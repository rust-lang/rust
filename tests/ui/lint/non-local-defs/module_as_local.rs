//! This test checks that module are treated as if they were local
//!
//! https://github.com/rust-lang/rust/issues/124396

//@ check-pass

trait JoinTo {}

fn simple_one() {
    mod posts {
        #[allow(non_camel_case_types)]
        pub struct table {}
    }

    impl JoinTo for posts::table {}
}

fn simple_two() {
    mod posts {
        pub mod posts {
            #[allow(non_camel_case_types)]
            pub struct table {}
        }
    }

    impl JoinTo for posts::posts::table {}
}

struct Global;
fn trait_() {
    mod posts {
        pub trait AdjecentTo {}
    }

    impl posts::AdjecentTo for Global {}
}

fn main() {}
