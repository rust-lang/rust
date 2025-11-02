//@ build-pass
//@ dont-check-compiler-stderr
//@ compile-flags:-Cremark=all -Cdebuginfo=1

fn main() {
    _ = "".split('.');
}
