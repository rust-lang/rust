// rustfmt-reorder_imports: true
// rustfmt-reorder_imported_names: true

use std::cmp::{a, b, c, d};
use std::ddd::{a, b, c as g, d as p};
use std::ddd::aaa;
// This comment should stay with `use std::ddd:bbb;`
use std::ddd::bbb;
/// This comment should stay with `use std::str;`
use std::str;
