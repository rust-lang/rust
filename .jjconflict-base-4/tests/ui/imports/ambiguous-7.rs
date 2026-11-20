// https://github.com/rust-lang/rust/pull/113099#issuecomment-1638206152

mod t2 {
    #[derive(Debug)]
    pub enum Error {}
}

pub use t2::*;

mod t3 {
    pub trait Error {}
}

use self::t3::*;
fn a<E: Error>(_: E) {}
//~^ ERROR `Error` is ambiguous

fn main() {}
