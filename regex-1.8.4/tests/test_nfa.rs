#![cfg_attr(feature = "pattern", feature(pattern))]

macro_rules! regex_new {
    ($re:expr) => {{
        use regex::internal::ExecBuilder;
        ExecBuilder::new($re).nfa().build().map(|e| e.into_regex())
    }};
}

macro_rules! regex {
    ($re:expr) => {
        regex_new!($re).unwrap()
    };
}

macro_rules! regex_set_new {
    ($re:expr) => {{
        use regex::internal::ExecBuilder;
        ExecBuilder::new_many($re).nfa().build().map(|e| e.into_regex_set())
    }};
}

macro_rules! regex_set {
    ($res:expr) => {
        regex_set_new!($res).unwrap()
    };
}

// Must come before other module definitions.
include!("macros_str.rs");
include!("macros.rs");

mod api;
mod api_str;
mod crazy;
mod flags;
mod fowler;
mod multiline;
mod noparse;
mod regression;
mod replace;
mod searcher;
mod set;
mod suffix_reverse;
#[cfg(feature = "unicode")]
mod unicode;
#[cfg(feature = "unicode-perl")]
mod word_boundary;
#[cfg(feature = "unicode-perl")]
mod word_boundary_unicode;
