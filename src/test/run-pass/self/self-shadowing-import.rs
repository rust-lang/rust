// run-pass

mod a {
    pub mod b {
        pub mod a {
            pub fn foo() -> isize { return 1; }
        }
    }
}

mod c {
    use a::b::a;
    pub fn bar() { assert_eq!(a::foo(), 1); }
}

pub fn main() { c::bar(); }
