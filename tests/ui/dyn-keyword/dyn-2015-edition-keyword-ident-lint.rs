// Under the 2015 edition with the keyword_idents lint, `dyn` is not
// entirely acceptable as an identifier. We currently do not attempt
// to detect or fix uses of `dyn` under a macro. Since we are testing
// this file via `rustfix`, we want the rustfix output to be
// compilable; so the macros here carefully use `dyn` "correctly."
//
//@ edition:2015
//@ run-rustfix

#![allow(non_camel_case_types)]
#![deny(keyword_idents)]

mod outer_mod {
    pub mod dyn {
//~^ ERROR `dyn` is a keyword
//~| WARN this is accepted in the current edition
        pub struct dyn;
//~^ ERROR `dyn` is a keyword
//~| WARN this is accepted in the current edition
    }
}
use outer_mod::dyn::dyn;
//~^ ERROR `dyn` is a keyword
//~| WARN this is accepted in the current edition
//~| ERROR `dyn` is a keyword
//~| WARN this is accepted in the current edition

fn main() {
    match dyn { dyn => {} }
//~^ ERROR `dyn` is a keyword
//~| WARN this is accepted in the current edition
//~| ERROR `dyn` is a keyword
//~| WARN this is accepted in the current edition
    macro_defn::dyn();
//~^ ERROR `dyn` is a keyword
//~| WARN this is accepted in the current edition

    macro_defn::boxed();
}

mod macro_defn {
    use super::Trait;

    macro_rules! dyn {
//~^ ERROR `dyn` is a keyword
//~| WARN this is accepted in the current edition

        // Note that we do not lint nor fix occurrences under macros
        ($dyn:tt) => { (Box<dyn Trait>, Box<$dyn Trait>) }
    }

    pub fn dyn() -> ::outer_mod::dyn::dyn {
//~^ ERROR `dyn` is a keyword
//~| WARN this is accepted in the current edition
//~| ERROR `dyn` is a keyword
//~| WARN this is accepted in the current edition
//~| ERROR `dyn` is a keyword
//~| WARN this is accepted in the current edition
        ::outer_mod::dyn::dyn
//~^ ERROR `dyn` is a keyword
//~| WARN this is accepted in the current edition
//~| ERROR `dyn` is a keyword
//~| WARN this is accepted in the current edition
    }



    pub fn boxed() -> dyn!(
        //~^ ERROR `dyn` is a keyword
        //~| WARN this is accepted in the current edition

            // Note that we do not lint nor fix occurrences under macros
            dyn
    )
    {
        (Box::new(1), Box::new(2))
    }
}

pub trait Trait { }

impl Trait for u32 { }
