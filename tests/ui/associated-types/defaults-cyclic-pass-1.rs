//@ check-pass

#![feature(associated_type_defaults)]

// Having a cycle in assoc. type defaults is okay, as long as there's no impl
// that retains it.
trait Tr {
    type A = Self::B;
    type B = Self::A;

    fn f();
}

// An impl has to break the cycle to be accepted.
impl Tr for u8 {
    type A = u8;

    fn f() {
        // Check that the type propagates as expected (seen from inside the impl)
        let _: Self::A = 0u8;
        let _: Self::B = 0u8;
    }
}

impl Tr for String {
    type B = ();

    fn f() {
        // Check that the type propagates as expected (seen from inside the impl)
        let _: Self::A = ();
        let _: Self::B = ();
    }
}

impl Tr for () {
    type A = Vec<()>;
    type B = u8;

    fn f() {
        // Check that the type propagates as expected (seen from inside the impl)
        let _: Self::A = Vec::<()>::new();
        let _: Self::B = 0u8;
    }
}

fn main() {
    // Check that both impls now have the right types (seen from outside the impls)
    let _: <u8 as Tr>::A = 0u8;
    let _: <u8 as Tr>::B = 0u8;

    let _: <String as Tr>::A = ();
    let _: <String as Tr>::B = ();

    let _: <() as Tr>::A = Vec::<()>::new();
    let _: <() as Tr>::B = 0u8;
}
