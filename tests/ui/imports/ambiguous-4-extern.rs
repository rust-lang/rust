//@ check-pass
// https://github.com/rust-lang/rust/pull/112743#issuecomment-1601986883

#![warn(ambiguous_glob_imports)]

macro_rules! m {
    () => {
      pub fn id() {}
    };
}

pub use evp::*; //~ WARNING ambiguous glob re-exports
pub use handwritten::*;

mod evp {
    use *;
    m! {}
}
mod handwritten {
    use *;
    m! {}
}

fn main() {
    id();
    //~^ WARNING `id` is ambiguous
    //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}
