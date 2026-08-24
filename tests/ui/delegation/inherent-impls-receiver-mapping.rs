#![feature(fn_delegation)]

mod receiver_mapping {
    struct X;

    impl X {
        fn static_f() {}
        fn by_value(self) {}
        fn by_ref(&self) {}
        fn by_mut_ref(&mut self) {}
    }

    struct Y;

    impl Y {
        fn get_x(&self) -> X { X }
        reuse X::{static_f, by_value, by_ref, by_mut_ref} { self.get_x() }
        //~^ ERROR: cannot find function `by_mut_ref` in `X`
        //~| ERROR: cannot find function `by_ref` in `X`
        //~| ERROR: cannot find function `by_value` in `X`
        //~| ERROR: cannot find function `static_f` in `X`
    }

    fn check() {
        let y = Y;
        y.by_ref();
        //~^ ERROR: no method named `by_ref` found for struct `Y` in the current scope
        y.by_mut_ref();
        //~^ ERROR: no method named `by_mut_ref` found for struct `Y` in the current scope
        y.by_value();
        //~^ ERROR: no method named `by_value` found for struct `Y` in the current scope

        let y = &Y;
        y.by_value();
        //~^ ERROR: no method named `by_value` found for reference `&Y` in the current scope
        y.by_ref();
        //~^ ERROR: no method named `by_ref` found for reference `&Y` in the current scope
        y.by_mut_ref();
        //~^ ERROR: no method named `by_mut_ref` found for reference `&Y` in the current scope

        let y = &mut Y;
        y.by_value();
        //~^ ERROR: no method named `by_value` found for mutable reference `&mut Y` in the current scope
        y.by_ref();
        //~^ ERROR: the method `by_ref` exists for mutable reference `&mut Y`, but its trait bounds were not satisfied
        y.by_mut_ref();
        //~^ ERROR: no method named `by_mut_ref` found for mutable reference `&mut Y` in the current scope
    }
}

mod self_type_mapping {
    struct X;
    impl X {
        fn add(self, other: Self) -> Self {
            Self
        }
    }

    struct W(X);
    impl W {
        reuse X::add { self.0 }
        //~^ ERROR: cannot find function `add` in `X`
    }

    fn check() {
        W(X).add(W(X));
        //~^ ERROR: no method named `add` found for struct `W` in the current scope
    }
}

fn main() {}
