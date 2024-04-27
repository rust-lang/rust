use a::{item /* comment */};
use b::{
    a,
    // comment
    item,
};
use c::item /* comment */;
use d::item; // really long comment (with `use` exactly 100 characters) ____________________________

use std::e::{/* it's a comment! */ bar /* and another */};
use std::f::{/* it's a comment! */ bar};
use std::g::{bar /* and another */};
