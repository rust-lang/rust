// aux-build:a.rs
// aux-build:b.rs
// revisions:rpass1 rpass2

#![feature(rustc_attrs)]


#[cfg(rpass1)]
extern crate a;
#[cfg(rpass1)]
extern crate b;

#[cfg(rpass2)]
extern crate b;
#[cfg(rpass2)]
extern crate a;

use a::A;
use b::B;

//? #[rustc_clean(label="typeck_tables_of", cfg="rpass2")]
pub fn main() {
    A + B;
}
