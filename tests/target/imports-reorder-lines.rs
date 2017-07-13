// rustfmt-reorder_imports: true

use std::cmp::{a, b, c, d};
use std::cmp::{b, e, f, g};
// This comment should stay with `use std::ddd;`
use std::ddd;
use std::ddd::aaa;
use std::ddd::bbb;
use std::str;

mod test {}

use aaa;
use aaa::*;
use aaa::bbb;

mod test {}
// If item names are equal, order by rename

use test::{a as aa, c};
use test::{a as bb, b};

mod test {}
// If item names are equal, order by rename - no rename comes before a rename

use test::{a, c};
use test::{a as bb, b};

mod test {}
// `self` always comes first

use test::{self as bb, b};
use test::{a as aa, c};
