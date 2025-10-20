//@ check-fail
//@ edition: 2024
#![deny(rust_2024_incompatible_pat)]

fn main() {}

#[derive(Copy, Clone)]
struct T;

struct Foo {
    f: &'static (u8,),
}

macro_rules! test_pat_on_type {
    ($($tt:tt)*) => {
        const _: () = {
            // Define a new function to ensure all cases are tested independently.
            fn foo($($tt)*) {}
        };
    };
}

test_pat_on_type![(&x,): &(T,)]; //~ ERROR mismatched types
test_pat_on_type![(&x,): &(&T,)]; //~ ERROR cannot explicitly dereference within an implicitly-borrowing pattern
test_pat_on_type![(&x,): &(&mut T,)]; //~ ERROR mismatched types
test_pat_on_type![(&mut x,): &(&T,)]; //~ ERROR mismatched types
test_pat_on_type![(&mut x,): &(&mut T,)]; //~ ERROR cannot explicitly dereference within an implicitly-borrowing pattern
test_pat_on_type![(&x,): &&mut &(T,)]; //~ ERROR mismatched types
test_pat_on_type![Foo { f: (&x,) }: Foo]; //~ ERROR mismatched types
test_pat_on_type![Foo { f: (&x,) }: &mut Foo]; //~ ERROR mismatched types
test_pat_on_type![Foo { f: &(x,) }: &Foo]; //~ ERROR cannot explicitly dereference within an implicitly-borrowing pattern
test_pat_on_type![(mut x,): &(T,)]; //~ ERROR cannot mutably bind by value within an implicitly-borrowing pattern
test_pat_on_type![(ref x,): &(T,)]; //~ ERROR cannot explicitly borrow within an implicitly-borrowing pattern
test_pat_on_type![(ref mut x,): &mut (T,)]; //~ ERROR cannot explicitly borrow within an implicitly-borrowing pattern

fn get<X>() -> X {
    unimplemented!()
}

// Make sure this works even when the underlying type is inferred. This test passes on rust stable.
fn infer<X: Copy>() -> X {
    match &get() {
        (&x,) => x, //~ ERROR cannot explicitly dereference within an implicitly-borrowing pattern
    }
}
