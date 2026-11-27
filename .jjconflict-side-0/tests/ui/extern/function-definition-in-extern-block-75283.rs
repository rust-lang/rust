//@ reference: items.extern.fn.body
// https://github.com/rust-lang/rust/issues/75283
extern "C" {
    fn lol() { //~ ERROR incorrect function inside `extern` block
        println!("");
    }
}
fn main() {}
