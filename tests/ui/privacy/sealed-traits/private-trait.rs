pub mod a {
    mod b {
        pub trait Hidden {}
    }
}

struct S;
impl a::b::Hidden for S {} //~ ERROR module `b` is private

fn main() {}
