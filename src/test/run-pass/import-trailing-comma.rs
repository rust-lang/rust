use foo::bar::{baz, quux,};

mod foo {
    #[legacy_exports];
    mod bar {
        #[legacy_exports];
        fn baz() { }
        fn quux() { }
    }
}

fn main() { baz(); quux(); }
