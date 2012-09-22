mod a {
    #[legacy_exports];
    mod b {
        #[legacy_exports];
        mod a {
            #[legacy_exports];
            fn foo() -> int { return 1; }
        }
    }
}

mod c {
    #[legacy_exports];
    use a::b::a;
    fn bar() { assert (a::foo() == 1); }
}

fn main() { c::bar(); }
