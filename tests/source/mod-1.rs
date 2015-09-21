// Deeply indented modules.

  mod foo { mod bar { mod baz {} } }

mod foo {
    mod bar {
    mod baz {
    fn foo() { bar() }
    }
    }

    mod qux {

    }
}

mod boxed { pub use std::boxed::{Box, HEAP}; }
