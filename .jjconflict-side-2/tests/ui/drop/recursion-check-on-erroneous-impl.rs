// can't use build-fail, because this also fails check-fail, but
// the ICE from #120787 only reproduces on build-fail.
//@ compile-flags: --emit=mir

struct PrintOnDrop<'a>(&'a str);

impl Drop for PrintOnDrop<'_> {
    fn drop() {} //~ ERROR method `drop` has a `&mut self` declaration in the trait
}

fn main() {}
