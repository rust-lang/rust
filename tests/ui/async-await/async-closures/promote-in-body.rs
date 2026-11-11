//@ build-pass
//@ compile-flags: --crate-type=lib
//@ edition: 2024

union U {
    f: i32,
}

fn foo() {
    async || {
        &U { f: 1 }
    };
}
