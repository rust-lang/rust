// https://github.com/rust-lang/rust/issues/33293
fn main() {
    match 0 {
        aaa::bbb(_) => ()
        //~^ ERROR failed to resolve: use of unresolved module or unlinked crate `aaa`
    };
}
