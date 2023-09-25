#![warn(clippy::wildcard_imports)]

mod prelude {
    pub const FOO: u8 = 1;
}
use prelude::*;
//~^ ERROR: usage of wildcard import

fn main() {
    let _ = FOO;
}
