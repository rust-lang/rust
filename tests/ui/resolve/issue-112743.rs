// https://github.com/rust-lang/rust/pull/112743#issuecomment-1601986883

macro_rules! m {
    () => {
      pub fn EVP_PKEY_id() {}
    };
}

mod openssl {
    pub use self::evp::*;
    pub use self::handwritten::*;

    mod evp {
      m!();
    }

    mod handwritten {
      m!();
    }
}
use openssl::*;

fn main() {
    EVP_PKEY_id(); //~ ERROR `EVP_PKEY_id` is ambiguous
}
