#![allow(non_upper_case_globals)]

use rustc_span::symbol::{Symbol, PREINTERNED_SYMBOLS_COUNT};

pub use rustc_span::sym::*;

macro_rules! generate {
    ($($sym:ident,)*) => {
        /// To be supplied to [`rustc_interface::Config`]
        pub const EXTRA_SYMBOLS: &[&str] = &[
            $(stringify!($sym),)*
        ];

        $(
            pub const $sym: Symbol = Symbol::new(PREINTERNED_SYMBOLS_COUNT + ${index()});
        )*
    };
}

generate! {
    rustfmt_skip,
    unused_extern_crates,
}
