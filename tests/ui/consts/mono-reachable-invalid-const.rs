//@ build-fail
//@ compile-flags: -Zdeduplicate-diagnostics=yes

struct Bar<const BITS: usize>;

impl<const BITS: usize> Bar<BITS> {
    const ASSERT: bool = {
        let b = std::convert::identity(1);
        ["oops"][b]; //~ ERROR evaluation of `Bar::<0>::ASSERT` failed
        true
    };

    fn assert() {
        let val = Self::ASSERT;
        if val {
            std::convert::identity(val);
        }
    }
}


fn main() {
    Bar::<0>::assert();
}
