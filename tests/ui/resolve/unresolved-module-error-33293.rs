// https://github.com/rust-lang/rust/issues/33293
fn main() {
    match 0 {
        aaa::bbb(_) => ()
        //~^ ERROR: cannot find module or crate `aaa`
    };
}
