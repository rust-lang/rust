// pretty-expanded FIXME #23616

mod a {
    pub mod b {
        pub type t = isize;

        pub fn foo() { let _x: t = 10; }
    }
}

pub fn main() { }
