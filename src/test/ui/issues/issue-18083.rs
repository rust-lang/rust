// check-pass
#![allow(dead_code)]
#![allow(unused_imports)]
// These crossed imports should resolve fine, and not block on
// each other and be reported as unresolved.

mod a {
    use b::{B};
    pub use self::inner::A;

    mod inner {
        pub struct A;
    }
}

mod b {
    use a::{A};
    pub use self::inner::B;

    mod inner {
        pub struct B;
    }
}

fn main() {}
