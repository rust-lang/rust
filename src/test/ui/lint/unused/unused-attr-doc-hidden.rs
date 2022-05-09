#![deny(unused_attributes)]
#![crate_type = "lib"]
// run-rustfix

pub trait Trait {
    type It;
    const IT: ();
    fn it0();
    fn it1();
    fn it2();
}

pub struct Implementor;

impl Trait for Implementor {
    #[doc(hidden)]
    type It = ();
    //~^^ ERROR `#[doc(hidden)]` is ignored
    //~|  WARNING this was previously accepted

    #[doc(hidden)]
    const IT: () = ();
    //~^^ ERROR `#[doc(hidden)]` is ignored
    //~|  WARNING this was previously accepted

    #[doc(hidden, alias = "aka")]
    fn it0() {}
    //~^^ ERROR `#[doc(hidden)]` is ignored
    //~|  WARNING this was previously accepted

    #[doc(alias = "this", hidden,)]
    fn it1() {}
    //~^^ ERROR `#[doc(hidden)]` is ignored
    //~|  WARNING this was previously accepted

    #[doc(hidden, hidden)]
    fn it2() {}
    //~^^ ERROR `#[doc(hidden)]` is ignored
    //~|  WARNING this was previously accepted
    //~|  ERROR `#[doc(hidden)]` is ignored
    //~|  WARNING this was previously accepted
}
