# SIAP

Predlog projekta iz predmeta Sistemi za istraživanje i analizu podataka
Uvod
Ovaj dokument sadrži kratak opis onoga što je tema i definicija projekta iz predmeta Sistemi za istraživanje i analizu podataka. Prikazana je motivacija za odabranu temu, zatim je dat kratak pregled dva relevantna naučna rada, kao i skup podataka koji će se koristiti u izradi projekta. Zatim je opisan metod evoluacije i naveden softver koji će se koristiti. Na kraju je prikazan plan izrade projekta.
Definicija projekta
Analizom rezimea filmova sa sajta IMDB utvrditi kojem žanru pripada film za koji je rezime pisan. Time bi se omogućilo da se za svaki film automatski odredi kojim žanrovima pripada, što bi kasnije omogućilo lakšu pretragu po žanrovima.
Motivacija
Internet Movie Database (IMDb) nalazi se na adresi www.imdb.com, a radi se o najvećoj bazi podataka o filmu na internetu koja sadrži mnoštvo podataka o filmovima, glumcima, rediteljima, scenaristima, producentima itd. Za svaki od filmova prikazan je kratak rezime koji opisuje radnju filma. Na osnovu njega čovek  može da prepozna kojim žanrovima pripada taj film. Ljudi najčešće čitaju rezime pre gledanja filma i vrše pretragu filmova po žanrovima, na osnovu svojih interesovanja. Na osnovu rečenica u rezimeu moguće je odrediti žanr kao što je akcija, komedija, horor, drama itd. Baš zato su i rezimei pisani na taj način da prenose informacije o žanrovima ljudima. Ako se za svaki film odredi kojim žanrovima pripada, kasnije je moguće preporučiti korisniku filmove koji su istog žanra ili tematike kao filmovi koje je radije gledao. Projekat bi automatski određivao žanr filma i na taj način bi se olakšalo određivanje žanra, bez učešća čoveka. 

Pregled vladajućih stavova i shvatanja o literaturi
[1] Ka Wing Ho (2011) Movies’ Genres Classification by Synopsis
http://cs229.stanford.edu/proj2011/Ho-MoviesGenresClassificationBySynopsis.pdf
Tema rada: Istražuje različite metode za klasifikaciju žanrova filmova na osnovu rezimea. Zadatak ovog projekta je klasifikacija sa više oznaka iz razloga što jedan film može u sebe uključivati više žanrova.
Korišteni algoritmi: One-vs-All SVM, K-nearest neighbour (KNN), Neuronska mreža. Sve ove metode koriste TF-IDF (Term frequency-Inverse document frequency) kao obilježja. Tekst fajl u vidu rezimea je procesiran da generiše Bag-Of-Words. Lista NLTK stop riječi je prvenstveno korištena da ukloni sve riječi koje se često pojavljuju, a nisu bitne za kontekst. Sve numeričke riječi su mapirane na isti indeks u riječniku iz razloga što ne obezbjeđuju mnogo informacija vezanih za žanr. Ostatak riječi je isfilterovan kroz Python matičnu biblioteku, te su iste riječi različitih oblika takođe mapirane na isti indeks u riječniku.
Podaci: Skup podataka koji je korišten je relativno mali sa 16000 unikatnih naslova, uključujući i rezime i žanr informacije što je podijeljeno na 80% i 20% redom na trening i test skup. Eksperiment je ograničen na predikciju samo 10 najpopularnijih žanrova. Imena žanrova i procenti filmova koji ih uključuju su:
Akcija (13.2%), Avantura (9.1%), Komedija (30%), Krimi (13.1%), Dokumentarni (16.3%), Drama (49.1%), Porodični (11%), Romantični (15.65%), Kratki filmovi (33.4%), Trileri (13.6%).
Ostvareni rezultati: SVM je postigao najbolji rezultat od 0.55, zatim Neuronska mreža sa 0.52, a na posljednjem mjestu KNN sa 0.51.

[2] Eric Makita and Artem Lenskiy (2016) A multinominal probabilistic model for movie genre predictions
https://arxiv.org/pdf/1603.07849.pdf
Tema rada: Predikcija žanra filma na osnovu korisničkih ocjena. Ideja je ustanoviti da su korisnici pretežno doslijedni svom odabiru filmova i da više preferiraju određeni žanr u odnosu na ostale.
Podaci: Kao skup podataka korišten je MovieLens dataset. Sadrži 1 milion ocjena za 3952 filmova od strane 6040 korisnika. Svaki korisnik je ocijenio bar 20 filmova na skali od 1 do 5.
Algoritmi: Naivni Bajes za klasifikaciju teksta i multinominalni model događaja. 
Ostvareni rezultati: Postiglo se 70% tačnosti predikcije korištenjem samo 15% trening skupa. Stopa tačnosti predikcije žanra se povećava sa povećanjem trening skupa podataka.

[3] Xingyou Wang, Weijie Jiang, Zhiyong Luo (2016) Combination of Convolutional and Recurrent Neural Network for Sentiment Analysis of Short Texts
https://www.aclweb.org/anthology/C16-1229.pdf
Tema rada: Spoj arhitekture konvolutivne i rekurentne neuronske mreže korišćenjem prednosti lokalnih karakteristika koje generiše konvolutivna mreža i zavisnosti u udaljenim delovima teksta naučenih putem rekurentne mreže za analizu sentimenta kratkog tekta.
Podaci: Metoda je evaluirana na tri skupa podataka: MR – recenzije filmova (po jedna rečenica recenziji), SST1 (Stanford Sentiment Treebank) – ekstenzija na MR skup podataka sa 5 vrsta etiketa: veoma negativno, negativno, neutralno, pozitivno i veoma pozitivno, i SST2 – kao i SST1 samo bez neutralnih recenzija i sa binarnim etiketama.
Algoritmi: Konvolutivna i rekurentna neuronska mreža. U konvolutivnoj mreži primenjuju se konvolucije na prozor reči u rečenici da bi se napravile karakteristike. Na mapu ovakvih karakteristika se zatim primenjuje pooling operacija kako bi se najbitnija karakteristika istakla. Algoritmi rekurentne neuronske mreže su LSTM (čuva memoriju) i GRU (pamti zavisnosti razlicnitih vremenskih skala).
Ostvareni rezultati: Eksperimentalni rezultati pokazuju 82% tačnosti na MR skupu podataka, 51% na SST1 i 89% na SST2.

Skup podataka
Skup podataka će činiti izvor informacija dostupan na internetu. Skup podataka koji je predviđen za korišćenje je MPST: Movie Plot Synopses with Tags, dostupan na Kaggle sajtu: 
kaggle datasets download -d cryptexcode/mpst-movie-plot-synopses-with-tags . Ovaj skup podataka sadrži 14828 filmova, odnosno podataka o njihovim rezimeima i tagovima (uključujući i žanrove), što nam je i potrebno za izradu ovog projekta. Atributi sadržani u podacima su:
    1. imdb_id (identifikator filma u IMDB bazi)
    2. title (naziv filma)
    3. plot_synopsis (rezime filma)
    4. tags (tagovi dodeljeni filmu)
    5. Split (da li podatak pripada skupu za trening/test)
    6. synopsis_source (odakle je prikupljen rezime filma)

Softver
Za izradu projekta planirano je koristiti Python programski jezik i scikit-learn biblioteku.

Metodologija
U skupu podataka bitno nam je polje koje se odnosi na rezime filma, i taj tekst treba da se pripremi za ulaz u neuronsku mrežu. Ovo podrazumeva primenu pretprocesiranja teksta kao što je izbacivanje znakova interpunkcije i lematizacija reči odnosno njeno prebacivanje u zajednički oblik. Zatim će se raditi TF-IDF nad tekstom, odnosno dodela važnosti svakoj reči u odnosu na jedan primerak dokumenta (rezimea) u kolekciji dokumenata (korpusu). Zbog toga se planira odraditi klasifikacija i klasičnim algoritmima poput SVM, Naive Bayes-a i Random Forest-a. Ukoliko bude potrebe zbog neuronske mreže, radiće se i tokenizacija teksta.
Korpus ce se obraditi putem BERT i ELMO algoritmama za reprezentaciju reči.
Neuronsku mrežu će sigurno činiti konvolutivna mreža, ali ako ona ne bude pokazivala dovoljno dobre rezultate, model će se spojiti sa rekurentnom mrežom, kao što je navedeno u radu Combination of Convolutional and Recurrent Neural Network for Sentiment Analysis of Short Texts [3]. Konvolutivna mreža je dobra u izvlačenju lokalnih i dubokih karakteristika teksta, dok su rekurentne mreže vremenski-rekurzivne i mogu da nauče dugoročne zavisnosti u sekvencijalnim podacima (npr. udaljene reči koje ukazuju jedne na druge).

Izlaz iz mreže je vektor svih postojećih žanrova u našem skupu podataka (podatak koji se nalazi u tags polju) sa skalarom koji određuje verovatnoću te klase za dati primerak. Skup podataka biće podeljen na obučavajući i validacioni skup (odnos 80:20), te će se mreža obučavati na obučavajućem skupu, a evaluacija tačnosti modela izvršiće se na validacionom skupu. Za test podatke ćemo uzeti neke skupove rezimea sa IMDB sajta koji se ne nalazi u postojećem skupu podataka i pustićemo da model izvrši predikciju, te ćemo gledati meru poklapanja sa stvarnim žanrovima filmova tih rezimea.

Plan izrade projekta
Realizacija projekta bi trebala da sadrži sledeće  bitne tačke:
    • Prikupljanje podataka
    •  Transformacija podataka
    •  Kreiranje modela
    •   Analiza dobijenih rezultata
 
Članovi tima
    • Jelena Garić
    • Ana Tomić
    • Jovan Jovkić


