// Verify that we ban usage of `Self` as constructor from inner items.

struct S0<T>(T);

impl<T> S0<T> {
    fn foo() {
        const C: S0<u8> = Self(0);
        //~^ ERROR can't use generic parameters from outer function
        fn bar() -> Self {
            //~^ ERROR can't use generic parameters from outer function
            Self(0)
            //~^ ERROR can't use generic parameters from outer function
        }
    }
}

fn main() {}
