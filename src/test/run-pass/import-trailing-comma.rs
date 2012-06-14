import foo::bar::{baz, quux,};

mod foo {
    mod bar {
        fn baz() { }
        fn quux() { }
    }
}

fn main() { baz(); quux(); }
