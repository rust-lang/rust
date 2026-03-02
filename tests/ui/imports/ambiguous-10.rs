// https://github.com/rust-lang/rust/pull/113099#issuecomment-1637022296

mod a {
    pub enum Token {}
}

mod b {
    use crate::a::*;
    #[derive(Debug)]
    pub enum Token {}
}

use crate::a::*;
use crate::b::*;
fn c(_: Token) {}
//~^ ERROR `Token` is ambiguous
//~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
fn main() { }
