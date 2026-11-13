//@ check-pass
//@ edition:2024

mod a {
    mod b {
        mod c {
            pub struct E;
        }
        mod d {
            mod c {
                pub struct E;
            }
            mod d {
                #[derive(Debug)]
                pub struct E;
            }
            pub use c::*;
            use d::*;
        }
        use c::*;
        use d::*;
    }
}

fn main() {}
