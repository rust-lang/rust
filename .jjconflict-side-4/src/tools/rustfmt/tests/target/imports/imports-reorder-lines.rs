use std::cmp::{a, b, c, d};
use std::cmp::{b, e, f, g};
use std::ddd::aaa;
use std::str;
// This comment should stay with `use std::ddd;`
use std::ddd;
use std::ddd::bbb;

mod test {}

use aaa;
use aaa::bbb;
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

use test::{self as bb, b};
use test::{a as aa, c};
