// https://github.com/rust-lang/rust/issues/98467

mod a {
    pub fn foo() {}
}

mod b {
    pub fn foo() {}
}

mod f {
    pub use a::*;
    pub use b::*;
}

mod g {
    pub use a::*;
    pub use f::*;
}

fn main() {
    g::foo();
    //~^ ERROR `foo` is ambiguous
    //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}
