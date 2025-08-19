//@ check-pass
// https://github.com/rust-lang/rust/pull/112743#issuecomment-1601986883

#![warn(ambiguous_glob_imports)]

macro_rules! m {
    () => {
      pub fn id() {}
    };
}

mod openssl {
    pub use self::evp::*;
    //~^ WARNING ambiguous glob re-exports
    pub use self::handwritten::*;

    mod evp {
      m!();
    }

    mod handwritten {
      m!();
    }
}

pub use openssl::*;

fn main() {
    id();
    //~^ WARNING `id` is ambiguous
    //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}
