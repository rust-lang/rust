//@ run-pass

// Tests that the return type of trait methods is correctly normalized when
// checking that a method in an impl matches the trait definition when the
// return type involves a defaulted associated type.
// ie. the trait has a method with return type `-> Self::R`, and `type R = ()`,
// but the impl leaves out the return type (resulting in `()`).
// Note that specialization is not involved in this test; no items in
// implementations may be overridden. If they were, the normalization wouldn't
// happen.

#![feature(associated_type_defaults)]

macro_rules! overload {
    ($a:expr, $b:expr) => {
        overload::overload2($a, $b)
    };
    ($a:expr, $b:expr, $c:expr) => {
        overload::overload3($a, $b, $c)
    }
}

fn main() {
    let () = overload!(42, true);

    let r: f32 = overload!("Hello world", 13.0);
    assert_eq!(r, 13.0);

    let () = overload!(42, true, 42.5);

    let r: i32 = overload!("Hello world", 13.0, 42);
    assert_eq!(r, 42);
}

mod overload {
    /// This trait has an assoc. type defaulting to `()`, and a required method returning a value
    /// of that assoc. type.
    pub trait Overload {
        // type R;
        type R = ();
        fn overload(self) -> Self::R;
    }

    // overloads for 2 args
    impl Overload for (i32, bool) {
        // type R = ();

        /// This function has no return type specified, and so defaults to `()`.
        ///
        /// This should work, but didn't, until RFC 2532 was implemented.
        fn overload(self) /*-> Self::R*/ {
            let (a, b) = self; // destructure args
            println!("i32 and bool {:?}", (a, b));
        }
    }
    impl<'a> Overload for (&'a str, f32) {
        type R = f32;
        fn overload(self) -> Self::R {
            let (a, b) = self; // destructure args
            println!("&str and f32 {:?}", (a, b));
            b
        }
    }

    // overloads for 3 args
    impl Overload for (i32, bool, f32) {
        // type R = ();
        fn overload(self) /*-> Self::R*/ {
            let (a, b, c) = self; // destructure args
            println!("i32 and bool and f32 {:?}", (a, b, c));
        }
    }
    impl<'a> Overload for (&'a str, f32, i32) {
        type R = i32;
        fn overload(self) -> Self::R {
            let (a, b, c) = self; // destructure args
            println!("&str and f32 and i32: {:?}", (a, b, c));
            c
        }
    }

    // overloads for more args
    // ...

    pub fn overload2<R, A, B>(a: A, b: B) -> R where (A, B): Overload<R = R> {
        (a, b).overload()
    }

    pub fn overload3<R, A, B, C>(a: A, b: B, c: C) -> R where (A, B, C): Overload<R = R> {
        (a, b, c).overload()
    }
}
