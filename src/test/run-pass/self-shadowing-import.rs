mod a {
    mod b {
        mod a {
            fn foo() -> int { return 1; }
        }
    }
}

mod c {
    import a::b::a;
    fn bar() { assert (a::foo() == 1); }
}

fn main() { c::bar(); }
