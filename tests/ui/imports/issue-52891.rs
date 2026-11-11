//@ aux-build:issue-52891.rs
//@ run-rustfix

#![allow(warnings)]

extern crate issue_52891;

// Check that we don't suggest renaming duplicate imports but instead
// suggest removing one.

use issue_52891::a;
use issue_52891::a; //~ ERROR `a` is defined multiple times

use issue_52891::{a, b, c}; //~ ERROR `a` is defined multiple times
use issue_52891::{d, a, e}; //~ ERROR `a` is defined multiple times
use issue_52891::{f, g, a}; //~ ERROR `a` is defined multiple times

use issue_52891::{a, //~ ERROR `a` is defined multiple times
    h,
    i};
use issue_52891::{j,
    a, //~ ERROR `a` is defined multiple times
    k};
use issue_52891::{l,
    m,
    a}; //~ ERROR `a` is defined multiple times

use issue_52891::a::inner;
use issue_52891::b::inner; //~ ERROR `inner` is defined multiple times

use issue_52891::{self};
//~^ ERROR `issue_52891` is defined multiple times

use issue_52891::n;
#[macro_use]
use issue_52891::n; //~ ERROR `n` is defined multiple times

fn main() {}
