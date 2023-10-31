// run-rustfix

pub mod a {
    pub use self::b::Trait;
    mod b {
        pub trait Trait {}
    }
}

struct S;
impl a::b::Trait for S {} //~ ERROR module `b` is private

fn main() {}
