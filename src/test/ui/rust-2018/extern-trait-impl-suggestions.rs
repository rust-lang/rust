// edition:2018
// aux-build:baz.rs
// compile-flags:--extern baz

// Don't use anything from baz - making suggestions from it when the only reference to it
// is an `--extern` flag is what is tested by this test.

struct Local;

fn main() {
    let local = Local;
    local.extern_baz(); //~ ERROR no method named `extern_baz`
}
