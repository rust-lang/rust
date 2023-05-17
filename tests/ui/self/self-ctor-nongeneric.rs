// `Self` as a constructor is currently allowed when the outer item is not generic.
// check-pass

struct S0(usize);

impl S0 {
    fn foo() {
        const C: S0 = Self(0);
        fn bar() -> S0 {
            Self(0)
        }
    }
}

fn main() {}
