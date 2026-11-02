//! `Pattern` must be an `unsafe trait`: `str::split_once` and friends slice the
//! string they were called on using indices from the searcher returned by
//! `Pattern::into_searcher`, without checking that the searcher searches that
//! same string. A safe impl could return a searcher over a different string,
//! which used to cause UB in safe code (see tracking issue #27721).
#![feature(pattern)]

use std::str::pattern::{CharSearcher, Pattern};

struct Bad;

impl Pattern for Bad {
    //~^ ERROR the trait `Pattern` requires an `unsafe impl` declaration
    type Searcher<'a> = CharSearcher<'a>;

    fn into_searcher<'a>(self, _haystack: &'a str) -> CharSearcher<'a> {
        'X'.into_searcher("aaaX")
    }
}

fn main() {
    let _ = "a".split_once(Bad);
}
