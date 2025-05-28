//@ build-fail

struct Bar<const BITS: usize>;

impl<const BITS: usize> Bar<BITS> {
    const ASSERT: bool = {
        let b = std::convert::identity(1);
        ["oops"][b]; //~ ERROR index out of bounds: the length is 1 but the index is 1
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
