//@ edition:2024

#![feature(fn_delegation)]
#![feature(type_info)]

mod ice_156342 {
    use std::mem::type_info::Trait;

    impl Trait {
        //~^ ERROR: cannot define inherent `impl` for a type outside of the crate where the type is defined
        reuse None::<&()>;
        //~^ ERROR: expected function, found unit variant `None`
    }

    fn foo<T>() {}

    reuse foo::<&&&&&&&&&&()> as foo1;
    reuse foo::<&std::borrow::Cow<'_, &()>> as foo2;
}

mod ice_156758 {
    trait X {}
    type Project = ();
    type Ty = ();

    impl X { //~ ERROR: expected a type, found a trait
        reuse<<<&Project> :: Ty> :: Ty as Iterator>::next;
        //~^ ERROR: ambiguous associated type
    }
}

fn main() {}
