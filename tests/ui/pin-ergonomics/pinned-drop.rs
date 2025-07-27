//@ edition:2024
#![feature(pin_ergonomics, negative_impls)]
#![allow(incomplete_features)]

macro_rules! def {
    ($name:ident: !Unpin) => {
        struct $name;
        impl !Unpin for $name {}
    };
    ($name:ident: ?Unpin) => {
        struct $name(std::marker::PhantomPinned);
    };
    ($name:ident: Unpin) => {
        struct $name;
        const _: crate::AssertUnpin<$name> = crate::AssertUnpin($name);
    };
}

macro_rules! impl_drop {
    ($name:ident: $($impls:tt)*) => {
        impl Drop for $name {
            $($impls)*
        }
    };
}

struct AssertUnpin<T: Unpin>(T);

mod drop {
    mod mut_self {
        def!(Foo: !Unpin);
        def!(Bar: ?Unpin);
        def!(Baz:  Unpin);

        impl_drop!(Foo: fn drop(&mut self) {}); //~ ERROR could not impl `Drop::drop(&mut self)` for a type that implements `!Unpin`
        impl_drop!(Bar: fn drop(&mut self) {}); // ok
        impl_drop!(Baz: fn drop(&mut self) {}); // ok
    }

    mod pin_mut_self {
        use std::pin::Pin;

        def!(Foo: !Unpin);
        def!(Bar: ?Unpin);
        def!(Baz:  Unpin);

        impl_drop!(Foo: fn drop(self: Pin<&mut Self>) {}); //~ ERROR method `drop` has an incompatible type for trait [E0053]
        //~^ ERROR could not impl `Drop::drop(&mut self)` for a type that implements `!Unpin`
        impl_drop!(Bar: fn drop(self: Pin<&mut Self>) {}); //~ ERROR method `drop` has an incompatible type for trait [E0053]
        impl_drop!(Baz: fn drop(self: Pin<&mut Self>) {}); //~ ERROR method `drop` has an incompatible type for trait [E0053]
    }

    mod pin_mut_self_sugar {
        def!(Foo: !Unpin);
        def!(Bar: ?Unpin);
        def!(Baz:  Unpin);

        impl_drop!(Foo: fn drop(&pin mut self) {}); //~ ERROR method `drop` has an incompatible type for trait [E0053]
        //~^ ERROR could not impl `Drop::drop(&mut self)` for a type that implements `!Unpin`
        impl_drop!(Bar: fn drop(&pin mut self) {}); //~ ERROR method `drop` has an incompatible type for trait [E0053]
        impl_drop!(Baz: fn drop(&pin mut self) {}); //~ ERROR method `drop` has an incompatible type for trait [E0053]
    }
}

mod pin_drop {
    mod mut_self {
        def!(Foo: !Unpin);
        def!(Bar: ?Unpin);
        def!(Baz:  Unpin);

        impl_drop!(Foo: fn pin_drop(&mut self) {}); //~ ERROR method `pin_drop` has an incompatible type for trait [E0053]
        impl_drop!(Bar: fn pin_drop(&mut self) {}); //~ ERROR method `pin_drop` has an incompatible type for trait [E0053]
        //~^ ERROR implementing `Drop::pin_drop(&pin mut self)` requires `Self: !Unpin`
        impl_drop!(Baz: fn pin_drop(&mut self) {}); //~ ERROR method `pin_drop` has an incompatible type for trait [E0053]
        //~^ ERROR implementing `Drop::pin_drop(&pin mut self)` requires `Self: !Unpin`
    }

    mod pin_mut_self {
        use std::pin::Pin;

        def!(Foo: !Unpin);
        def!(Bar: ?Unpin);
        def!(Baz:  Unpin);

        impl_drop!(Foo: fn pin_drop(self: Pin<&mut Self>) {}); // ok
        impl_drop!(Bar: fn pin_drop(self: Pin<&mut Self>) {}); //~ ERROR implementing `Drop::pin_drop(&pin mut self)` requires `Self: !Unpin`
        impl_drop!(Baz: fn pin_drop(self: Pin<&mut Self>) {}); //~ ERROR implementing `Drop::pin_drop(&pin mut self)` requires `Self: !Unpin`
    }

    mod pin_mut_self_sugar {
        def!(Foo: !Unpin);
        def!(Bar: ?Unpin);
        def!(Baz:  Unpin);

        impl_drop!(Foo: fn pin_drop(&pin mut self) {}); // ok
        impl_drop!(Bar: fn pin_drop(&pin mut self) {}); //~ ERROR implementing `Drop::pin_drop(&pin mut self)` requires `Self: !Unpin`
        impl_drop!(Baz: fn pin_drop(&pin mut self) {}); //~ ERROR implementing `Drop::pin_drop(&pin mut self)` requires `Self: !Unpin`
    }
}

mod both {
    def!(Foo: !Unpin);
    def!(Bar: ?Unpin);
    def!(Baz:  Unpin);

    impl Drop for Foo {
        //~^ ERROR conflict implementation of `Drop::drop` and `Drop::pin_drop`
        fn drop(&mut self) {}
        fn pin_drop(&pin mut self) {}
    }
    impl Drop for Bar {
        //~^ ERROR conflict implementation of `Drop::drop` and `Drop::pin_drop`
        fn drop(&mut self) {}
        fn pin_drop(&pin mut self) {}
    }
    impl Drop for Baz {
        //~^ ERROR conflict implementation of `Drop::drop` and `Drop::pin_drop`
        fn drop(&mut self) {}
        fn pin_drop(&pin mut self) {}
    }
}

mod empty {
    def!(Foo: !Unpin);
    def!(Bar: ?Unpin);
    def!(Baz:  Unpin);

    impl Drop for Foo {} //~ ERROR not all trait items implemented, missing one of: `drop`, `pin_drop` [E0046]
    impl Drop for Bar {} //~ ERROR not all trait items implemented, missing one of: `drop`, `pin_drop` [E0046]
    impl Drop for Baz {} //~ ERROR not all trait items implemented, missing one of: `drop`, `pin_drop` [E0046]
}

mod irrelevant {
    def!(Foo: !Unpin);
    def!(Bar: ?Unpin);
    def!(Baz:  Unpin);

    impl Drop for Foo {
        //~^ ERROR not all trait items implemented, missing one of: `drop`, `pin_drop`
        type Type = ();     //~ ERROR type `Type` is not a member of trait `Drop` [E0437]
        const N: usize = 0; //~ ERROR const `N` is not a member of trait `Drop`
        fn foo() {}         //~ ERROR method `foo` is not a member of trait `Drop`
    }
    impl Drop for Bar {
        //~^ ERROR not all trait items implemented, missing one of: `drop`, `pin_drop`
        type Type = ();     //~ ERROR type `Type` is not a member of trait `Drop` [E0437]
        const N: usize = 0; //~ ERROR const `N` is not a member of trait `Drop`
        fn foo() {}         //~ ERROR method `foo` is not a member of trait `Drop`
    }
    impl Drop for Baz {
        //~^ ERROR not all trait items implemented, missing one of: `drop`, `pin_drop`
        type Type = ();     //~ ERROR type `Type` is not a member of trait `Drop` [E0437]
        const N: usize = 0; //~ ERROR const `N` is not a member of trait `Drop`
        fn foo() {}         //~ ERROR method `foo` is not a member of trait `Drop`
    }
}

mod explicit_call_pin_drop {
    def!(Foo: !Unpin);
    def!(Bar: ?Unpin);
    def!(Baz:  Unpin);

    impl_drop!(Foo: fn drop(&mut self) { Drop::pin_drop(todo!()) }); //~ ERROR explicit use of destructor method [E0040]
    //~^ ERROR could not impl `Drop::drop(&mut self)` for a type that implements `!Unpin`
    impl_drop!(Bar: fn drop(&mut self) { Drop::pin_drop(todo!()) }); //~ ERROR explicit use of destructor method [E0040]
    impl_drop!(Baz: fn drop(&mut self) { Drop::pin_drop(todo!()) }); //~ ERROR explicit use of destructor method [E0040]
}

mod explicit_call_drop {
    def!(Foo: !Unpin);
    def!(Bar: ?Unpin);
    def!(Baz:  Unpin);

    impl_drop!(Foo: fn pin_drop(&pin mut self) { Drop::drop(todo!()) }); //~ ERROR explicit use of destructor method [E0040]
    impl_drop!(Bar: fn pin_drop(&pin mut self) { Drop::drop(todo!()) }); //~ ERROR explicit use of destructor method [E0040]
    //~^ ERROR implementing `Drop::pin_drop(&pin mut self)` requires `Self: !Unpin`
    impl_drop!(Baz: fn pin_drop(&pin mut self) { Drop::drop(todo!()) }); //~ ERROR explicit use of destructor method [E0040]
    //~^ ERROR implementing `Drop::pin_drop(&pin mut self)` requires `Self: !Unpin`
}

fn main() {}
