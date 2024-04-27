//@ revisions: tuple unit struct_
//@[unit] check-pass

#[cfg(unit)]
mod unit {
    struct S;
    impl S {
        fn foo() {
            let Self = S;
        }
    }
}

#[cfg(tuple)]
mod tuple {
    struct S(());
    impl S {
        fn foo() {
            let Self = S;
            //[tuple]~^ ERROR expected unit struct
        }
    }
}

#[cfg(struct_)]
mod struct_ {
    struct S {}
    impl S {
        fn foo() {
            let Self = S;
            //[struct_]~^ ERROR expected value, found struct `S`
            //[struct_]~| ERROR expected unit struct, found self constructor `Self`
        }
    }
}

fn main() {}
