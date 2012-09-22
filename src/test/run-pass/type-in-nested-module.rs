

mod a {
    #[legacy_exports];
    mod b {
        #[legacy_exports];
        type t = int;

        fn foo() { let x: t = 10; }
    }
}

fn main() { }
