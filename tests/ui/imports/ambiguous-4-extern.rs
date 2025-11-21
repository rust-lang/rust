//@ edition:2015
// https://github.com/rust-lang/rust/pull/112743#issuecomment-1601986883

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
    //~^ ERROR `id` is ambiguous
}
