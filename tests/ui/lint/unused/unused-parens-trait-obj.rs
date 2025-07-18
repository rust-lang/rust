//@ revisions: edition2015 edition2018
//@[edition2015] check-pass
//@[edition2015] edition: 2015
//@[edition2018] run-rustfix
//@[edition2018] edition: 2018

#![deny(unused_parens)]

#[allow(unused)]
macro_rules! edition2015_only {
    () => {
        mod dyn {
            pub type IsAContextualKeywordIn2015 = ();
        }

        pub type DynIsAContextualKeywordIn2015A = dyn::IsAContextualKeywordIn2015;
    }
}

#[cfg(edition2015)]
edition2015_only!();

// there's a lint for 2018 and later only because of how dyn is parsed in edition 2015
//[edition2018]~v ERROR unnecessary parentheses around type
pub type DynIsAContextualKeywordIn2015B = Box<dyn (::std::ops::Fn())>;

fn main() {}
