use std::str;
use std::cmp::{d, c, b, a};
use std::cmp::{b, e, g, f};
use std::ddd::aaa;
// This comment should stay with `use std::ddd;`
use std::ddd;
use std::ddd::bbb;

mod test {
}

use aaa::bbb;
use aaa;
use aaa::*;

mod test {}
// If item names are equal, order by rename

use test::{a as bb, b};
use test::{a as aa, c};

mod test {}
// If item names are equal, order by rename - no rename comes before a rename

use test::{a as bb, b};
use test::{a, c};

mod test {}
// `self` always comes first

use test::{a as aa, c};
use test::{self as bb, b};
