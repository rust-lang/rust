#![crate_name = "globuliferous"]
#![feature(doc_cfg)]

//@ has 'globuliferous/index.html'
//@ count   - '//*[@class="stab portability"]' 1
//@ matches - '//*[@class="stab portability"]' '^ratel$'

//@ has 'globuliferous/ratel/index.html'
//@ count   - '//*[@class="stab portability"]' 8
//@ matches - '//*[@class="stab portability"]' 'crate feature ratel'
//@ matches - '//*[@class="stab portability"]' '^zoonosology$'
//@ matches - '//*[@class="stab portability"]' '^yusho$'
//@ matches - '//*[@class="stab portability"]' '^nunciative$'
//@ matches - '//*[@class="stab portability"]' '^thionic$'
//@ matches - '//*[@class="stab portability"]' '^zincic$'
//@ matches - '//*[@class="stab portability"]' '^cosmotellurian$'
//@ matches - '//*[@class="stab portability"]' '^aposiopesis$'
#[doc(cfg(feature = "ratel"))]
pub mod ratel {
    //@ has 'globuliferous/ratel/fn.ovicide.html'
    //@ count   - '//*[@class="stab portability"]' 1
    //@ matches - '//*[@class="stab portability"]' 'crate feature ratel'
    pub fn ovicide() {}

    //@ has 'globuliferous/ratel/fn.zoonosology.html'
    //@ count   - '//*[@class="stab portability"]' 1
    //@ matches - '//*[@class="stab portability"]' 'crate features ratel and zoonosology'
    #[doc(cfg(feature = "zoonosology"))]
    pub fn zoonosology() {}

    //@ has 'globuliferous/ratel/constant.DIAGRAPHICS.html'
    //@ count   - '//*[@class="stab portability"]' 1
    //@ matches - '//*[@class="stab portability"]' 'crate feature ratel'
    pub const DIAGRAPHICS: () = ();

    //@ has 'globuliferous/ratel/constant.YUSHO.html'
    //@ count   - '//*[@class="stab portability"]' 1
    //@ matches - '//*[@class="stab portability"]' 'crate features ratel and yusho'
    #[doc(cfg(feature = "yusho"))]
    pub const YUSHO: () = ();

    //@ has 'globuliferous/ratel/static.KEYBUGLE.html'
    //@ count   - '//*[@class="stab portability"]' 1
    //@ matches - '//*[@class="stab portability"]' 'crate feature ratel'
    pub static KEYBUGLE: () = ();

    //@ has 'globuliferous/ratel/static.NUNCIATIVE.html'
    //@ count   - '//*[@class="stab portability"]' 1
    //@ matches - '//*[@class="stab portability"]' 'crate features ratel and nunciative'
    #[doc(cfg(feature = "nunciative"))]
    pub static NUNCIATIVE: () = ();

    //@ has 'globuliferous/ratel/type.Wrick.html'
    //@ count   - '//*[@class="stab portability"]' 1
    //@ matches - '//*[@class="stab portability"]' 'crate feature ratel'
    pub type Wrick = ();

    //@ has 'globuliferous/ratel/type.Thionic.html'
    //@ count   - '//*[@class="stab portability"]' 1
    //@ matches - '//*[@class="stab portability"]' 'crate features ratel and thionic'
    #[doc(cfg(feature = "thionic"))]
    pub type Thionic = ();

    //@ has 'globuliferous/ratel/struct.Eventration.html'
    //@ count   - '//*[@class="stab portability"]' 1
    //@ matches - '//*[@class="stab portability"]' 'crate feature ratel'
    pub struct Eventration;

    //@ has 'globuliferous/ratel/struct.Zincic.html'
    //@ count   - '//*[@class="stab portability"]' 2
    //@ matches - '//*[@class="stab portability"]' 'crate features ratel and zincic'
    //@ matches - '//*[@class="stab portability"]' 'crate feature rutherford'
    #[doc(cfg(feature = "zincic"))]
    pub struct Zincic {
        pub rectigrade: (),

        #[doc(cfg(feature = "rutherford"))]
        pub rutherford: (),
    }

    //@ has 'globuliferous/ratel/enum.Cosmotellurian.html'
    //@ count   - '//*[@class="stab portability"]' 10
    //@ matches - '//*[@class="stab portability"]' 'crate features ratel and cosmotellurian'
    //@ matches - '//*[@class="stab portability"]' 'crate feature biotaxy'
    //@ matches - '//*[@class="stab portability"]' 'crate feature xiphopagus'
    //@ matches - '//*[@class="stab portability"]' 'crate feature juxtapositive'
    //@ matches - '//*[@class="stab portability"]' 'crate feature fuero'
    //@ matches - '//*[@class="stab portability"]' 'crate feature palaeophile'
    //@ matches - '//*[@class="stab portability"]' 'crate feature broadcloth'
    //@ matches - '//*[@class="stab portability"]' 'crate features broadcloth and xanthocomic'
    //@ matches - '//*[@class="stab portability"]' 'crate feature broadcloth'
    //@ matches - '//*[@class="stab portability"]' 'crate features broadcloth and whosoever'
    #[doc(cfg(feature = "cosmotellurian"))]
    pub enum Cosmotellurian {
        Groundsel {
            jagger: (),

            #[doc(cfg(feature = "xiphopagus"))]
            xiphopagus: (),
        },

        #[doc(cfg(feature = "biotaxy"))]
        Biotaxy {
            glossography: (),

            #[doc(cfg(feature = "juxtapositive"))]
            juxtapositive: (),
        },
    }

    impl Cosmotellurian {
        pub fn uxoricide() {}

        #[doc(cfg(feature = "fuero"))]
        pub fn fuero() {}

        pub const MAMELLE: () = ();

        #[doc(cfg(feature = "palaeophile"))]
        pub const PALAEOPHILE: () = ();
    }

    #[doc(cfg(feature = "broadcloth"))]
    impl Cosmotellurian {
        pub fn trabeculated() {}

        #[doc(cfg(feature = "xanthocomic"))]
        pub fn xanthocomic() {}

        pub const BRACHIFEROUS: () = ();

        #[doc(cfg(feature = "whosoever"))]
        pub const WHOSOEVER: () = ();
    }

    //@ has 'globuliferous/ratel/trait.Gnotobiology.html'
    //@ count   - '//*[@class="stab portability"]' 4
    //@ matches - '//*[@class="stab portability"]' 'crate feature ratel'
    //@ matches - '//*[@class="stab portability"]' 'crate feature unzymotic'
    //@ matches - '//*[@class="stab portability"]' 'crate feature summate'
    //@ matches - '//*[@class="stab portability"]' 'crate feature unctuous'
    pub trait Gnotobiology {
        const XYLOTHERAPY: ();

        #[doc(cfg(feature = "unzymotic"))]
        const UNZYMOTIC: ();

        type Lepadoid;

        #[doc(cfg(feature = "summate"))]
        type Summate;

        fn decalcomania();

        #[doc(cfg(feature = "unctuous"))]
        fn unctuous();
    }

    //@ has 'globuliferous/ratel/trait.Aposiopesis.html'
    //@ count   - '//*[@class="stab portability"]' 4
    //@ matches - '//*[@class="stab portability"]' 'crate features ratel and aposiopesis'
    //@ matches - '//*[@class="stab portability"]' 'crate feature umbracious'
    //@ matches - '//*[@class="stab portability"]' 'crate feature uakari'
    //@ matches - '//*[@class="stab portability"]' 'crate feature rotograph'
    #[doc(cfg(feature = "aposiopesis"))]
    pub trait Aposiopesis {
        const REDHIBITION: ();

        #[doc(cfg(feature = "umbracious"))]
        const UMBRACIOUS: ();

        type Ophthalmoscope;

        #[doc(cfg(feature = "uakari"))]
        type Uakari;

        fn meseems();

        #[doc(cfg(feature = "rotograph"))]
        fn rotograph();
    }
}
