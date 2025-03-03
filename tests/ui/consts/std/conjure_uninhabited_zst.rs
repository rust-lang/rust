#![feature(mem_conjure_zst)]

use std::convert::Infallible;
use std::mem::conjure_zst;

// not ok, since the type needs to be inhabited
const CONJURE_INVALID: Infallible = unsafe { conjure_zst() };
//~^ ERROR any use of this value will cause an error
//~^^ WARN will become a hard error in a future release

fn main() {}
