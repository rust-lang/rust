//@ edition:2015
// https://github.com/rust-lang/rust/pull/113099#issuecomment-1638206152

pub use evp::*; //~ WARNING ambiguous glob re-exports
pub use handwritten::*;

macro_rules! m {
    () => {
        pub fn id() {}
    };
}
mod evp {
    use *;
    m!();
}

mod handwritten {
    pub use handwritten::evp::*;
    mod evp {
        use *;
        m!();
    }
}

fn main() {
    id();
    //~^ ERROR `id` is ambiguous
}
