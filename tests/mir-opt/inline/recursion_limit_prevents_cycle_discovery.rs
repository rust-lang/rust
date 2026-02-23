//@ aux-build: wrapper.rs
//@ compile-flags: -Zmir-opt-level=2 -Zinline-mir
// skip-filecheck

// This is a regression test for https://github.com/rust-lang/rust/issues/146998

extern crate wrapper;
use wrapper::{Compare, wrap};

pub struct BundleInner;

impl Compare for BundleInner {
    fn eq(self) {
        lots_of_calls();
        wrap(Resource::ExtensionValue);
    }
}

pub enum Resource {
    Bundle,
    ExtensionValue,
}

impl Compare for Resource {
    fn eq(self) {
        match self {
            Self::Bundle => wrap(BundleInner),
            Self::ExtensionValue => lots_of_calls(),
        }
    }
}

macro_rules! units {
    ($($n: ident)*) => {
        $(
            struct $n;

            impl Compare for $n {
                fn eq(self) {  }
            }

            wrap($n);
        )*
    };
}

fn lots_of_calls() {
    units! {
        O1 O2 O3 O4 O5 O6 O7 O8 O9 O10 O11 O12 O13 O14 O15 O16 O17 O18 O19 O20
        O21 O22 O23 O24 O25 O26 O27 O28 O29 O30 O31 O32 O33 O34 O35 O36 O37 O38 O39 O40
        O41 O42 O43 O44 O45 O46 O47 O48 O49 O50 O51 O52 O53 O54 O55 O56 O57 O58 O59 O60
        O61 O62 O63 O64 O65
    }
}
