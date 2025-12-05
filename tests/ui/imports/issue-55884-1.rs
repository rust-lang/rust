mod m {
    mod m1 {
        pub struct S {}
    }
    mod m2 {
        // Note this derive, it makes this struct macro-expanded,
        // so it doesn't appear in time to participate in the initial resolution of `use m::S`,
        // only in the later validation pass.
        #[derive(Default)]
        pub struct S {}
    }

    // Create a glob vs glob ambiguity
    pub use self::m1::*;
    pub use self::m2::*;
}

fn main() {
    use m::S; //~ ERROR `S` is ambiguous
    let s = S {};
}
