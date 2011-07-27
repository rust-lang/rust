

mod a {
    mod b {
        type t = int;

        fn foo() { let x: t = 10; }
    }
}

fn main() { }