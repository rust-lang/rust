// Deeply indented modules.

mod foo {
    mod bar {
        mod baz {
            fn foo() {
                bar()
            }
        }
    }

    mod qux {

    }
}
