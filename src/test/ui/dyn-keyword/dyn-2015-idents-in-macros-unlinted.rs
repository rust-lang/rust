// compile-pass

// Under the 2015 edition with the keyword_idents lint, `dyn` is
// not entirely acceptable as an identifier.
//
// We currently do not attempt to detect or fix uses of `dyn` as an
// identifier under a macro.

#![allow(non_camel_case_types)]
#![deny(keyword_idents)]

mod outer_mod {
    pub mod r#dyn {
        pub struct r#dyn;
    }
}

macro_rules! defn_has_dyn_idents {
    ($arg:ident) => { ::outer_mod::dyn::dyn }
}

fn main() {
    defn_has_dyn_idents!(dyn);
}
