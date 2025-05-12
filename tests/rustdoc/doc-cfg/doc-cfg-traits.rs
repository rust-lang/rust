#![crate_name = "myrmecophagous"]
#![feature(doc_cfg, associated_type_defaults)]

//@ has 'myrmecophagous/index.html'
//@ count   - '//*[@class="stab portability"]' 2
//@ matches - '//*[@class="stab portability"]' '^jurisconsult$'
//@ matches - '//*[@class="stab portability"]' '^quarter$'

pub trait Lea {}

//@ has 'myrmecophagous/trait.Vortoscope.html'
//@ count   - '//*[@class="stab portability"]' 6
//@ matches - '//*[@class="stab portability"]' 'crate feature zibib'
//@ matches - '//*[@class="stab portability"]' 'crate feature poriform'
//@ matches - '//*[@class="stab portability"]' 'crate feature ethopoeia'
//@ matches - '//*[@class="stab portability"]' 'crate feature lea'
//@ matches - '//*[@class="stab portability"]' 'crate feature unit'
//@ matches - '//*[@class="stab portability"]' 'crate feature quarter'
pub trait Vortoscope {
    type Batology = ();

    #[doc(cfg(feature = "zibib"))]
    type Zibib = ();

    const YAHRZEIT: () = ();

    #[doc(cfg(feature = "poriform"))]
    const PORIFORM: () = ();

    fn javanais() {}

    #[doc(cfg(feature = "ethopoeia"))]
    fn ethopoeia() {}
}

#[doc(cfg(feature = "lea"))]
impl<T: Lea> Vortoscope for T {}

#[doc(cfg(feature = "unit"))]
impl Vortoscope for () {}

//@ has 'myrmecophagous/trait.Jurisconsult.html'
//@ count   - '//*[@class="stab portability"]' 7
//@ matches - '//*[@class="stab portability"]' 'crate feature jurisconsult'
//@ matches - '//*[@class="stab portability"]' 'crate feature lithomancy'
//@ matches - '//*[@class="stab portability"]' 'crate feature boodle'
//@ matches - '//*[@class="stab portability"]' 'crate feature mistetch'
//@ matches - '//*[@class="stab portability"]' 'crate feature lea'
//@ matches - '//*[@class="stab portability"]' 'crate feature unit'
//@ matches - '//*[@class="stab portability"]' 'crate feature quarter'
#[doc(cfg(feature = "jurisconsult"))]
pub trait Jurisconsult {
    type Urbanist = ();

    #[doc(cfg(feature = "lithomancy"))]
    type Lithomancy = ();

    const UNIFILAR: () = ();

    #[doc(cfg(feature = "boodle"))]
    const BOODLE: () = ();

    fn mersion() {}

    #[doc(cfg(feature = "mistetch"))]
    fn mistetch() {}
}

#[doc(cfg(feature = "lea"))]
impl<T: Lea> Jurisconsult for T {}

#[doc(cfg(feature = "unit"))]
impl Jurisconsult for () {}

//@ has 'myrmecophagous/struct.Ultimogeniture.html'
//@ count   - '//*[@class="stab portability"]' 8
//
//@ matches - '//*[@class="stab portability"]' 'crate feature zibib'
//@ matches - '//*[@class="stab portability"]' 'crate feature poriform'
//@ matches - '//*[@class="stab portability"]' 'crate feature ethopoeia'
//
//@ matches - '//*[@class="stab portability"]' 'crate feature jurisconsult'
//@ matches - '//*[@class="stab portability"]' 'crate feature lithomancy'
//@ matches - '//*[@class="stab portability"]' 'crate feature boodle'
//@ matches - '//*[@class="stab portability"]' 'crate feature mistetch'
//
//@ matches - '//*[@class="stab portability"]' 'crate feature copy'
#[derive(Clone)]
pub struct Ultimogeniture;

impl Vortoscope for Ultimogeniture {}

#[doc(cfg(feature = "jurisconsult"))]
impl Jurisconsult for Ultimogeniture {}

#[doc(cfg(feature = "copy"))]
impl Copy for Ultimogeniture {}

//@ has 'myrmecophagous/struct.Quarter.html'
//@ count   - '//*[@class="stab portability"]' 9
//@ matches - '//*[@class="stab portability"]' 'crate feature quarter'
//
//@ matches - '//*[@class="stab portability"]' 'crate feature zibib'
//@ matches - '//*[@class="stab portability"]' 'crate feature poriform'
//@ matches - '//*[@class="stab portability"]' 'crate feature ethopoeia'
//
//@ matches - '//*[@class="stab portability"]' 'crate feature jurisconsult'
//@ matches - '//*[@class="stab portability"]' 'crate feature lithomancy'
//@ matches - '//*[@class="stab portability"]' 'crate feature boodle'
//@ matches - '//*[@class="stab portability"]' 'crate feature mistetch'
//
//@ matches - '//*[@class="stab portability"]' 'crate feature copy'
#[doc(cfg(feature = "quarter"))]
#[derive(Clone)]
pub struct Quarter;

#[doc(cfg(feature = "quarter"))]
impl Vortoscope for Quarter {}

#[doc(cfg(all(feature = "jurisconsult", feature = "quarter")))]
impl Jurisconsult for Quarter {}

#[doc(cfg(all(feature = "copy", feature = "quarter")))]
impl Copy for Quarter {}
