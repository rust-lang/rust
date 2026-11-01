// Make sure we don't suggest a method change inside the `write!` macro.
//
// See <https://github.com/rust-lang/rust/issues/139830>

fn main() {
    let mut buf = String::new();
    let _ = write!(buf, "foo");
    //~^ ERROR cannot write into `String`
}
