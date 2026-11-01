struct S0<T>(T);

impl<T> S0<T> {
    fn foo() {
        const C: S0<i32> = Self(0);
        //~^ ERROR can't reference `Self` constructor from outer item
        fn bar() -> S0<i32> {
            Self(0)
            //~^ ERROR can't reference `Self` constructor from outer item
        }
    }
}

fn main() {}
