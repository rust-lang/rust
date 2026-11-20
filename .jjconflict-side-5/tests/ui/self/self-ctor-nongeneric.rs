// `Self` as a constructor is currently allowed when the outer item is not generic.
//@ check-pass

struct S0(usize);

impl S0 {
    fn foo() {
        const C: S0 = Self(0);
        //~^ WARN can't reference `Self` constructor from outer item
        //~| WARN this was previously accepted by the compiler but is being phased out
        fn bar() -> S0 {
            Self(0)
            //~^ WARN can't reference `Self` constructor from outer item
            //~| WARN this was previously accepted by the compiler but is being phased out
        }
    }
}

fn main() {}
