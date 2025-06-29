//@ check-pass
//@ edition: 2015

#![deny(unused_parens)]

mod dyn {
    pub type IsAContextualKeywordIn2015 = ();
}

pub type DynIsAContextualKeywordIn2015A = dyn::IsAContextualKeywordIn2015;

// there's no lint here, because the type of how dyn is parsed in edition 2015
pub type DynIsAContextualKeywordIn2015B = Box<dyn (::std::ops::Fn())>;

fn main() {}
