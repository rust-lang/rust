//@ run-pass
//@ check-run-results

// Tests ensuring that `dbg!(expr)` has the expected run-time behavior.
// as well as some compile time properties we expect.

#![allow(dropping_copy_types)]

#[derive(Copy, Clone, Debug)]
struct Unit;

#[derive(Copy, Clone, Debug, PartialEq)]
struct Point<T> {
    x: T,
    y: T,
}

#[derive(Debug, PartialEq)]
struct NoCopy(usize);

fn main() {
    let a: Unit = dbg!(Unit);
    let _: Unit = dbg!(a);
    // We can move `a` because it's Copy.
    drop(a);

    // `Point<T>` will be faithfully formatted according to `{:#?}`.
    let a = Point { x: 42, y: 24 };
    let b: Point<u8> = dbg!(Point { x: 42, y: 24 }); // test stringify!(..)
    let c: Point<u8> = dbg!(b);
    // Identity conversion:
    assert_eq!(a, b);
    assert_eq!(a, c);
    // We can move `b` because it's Copy.
    drop(b);

    // Without parameters works as expected.
    let _: () = dbg!();

    // Test that we can borrow and that successive applications is still identity.
    let a = NoCopy(1337);
    let b: &NoCopy = dbg!(dbg!(&a));
    assert_eq!(&a, b);

    // Test involving lifetimes of temporaries:
    fn f<'a>(x: &'a u8) -> &'a u8 { x }
    let a: &u8 = dbg!(f(&42));
    assert_eq!(a, &42);

    // Test side effects:
    let mut foo = 41;
    assert_eq!(7331, dbg!({
        foo += 1;
        eprintln!("before");
        7331
    }));
    assert_eq!(foo, 42);

    // Test trailing comma:
    assert_eq!(("Yeah",), dbg!(("Yeah",)));

    // Test multiple arguments:
    assert_eq!((1u8, 2u32), dbg!(1,
                                 2));

    // Test multiple arguments + trailing comma:
    assert_eq!((1u8, 2u32, "Yeah"), dbg!(1u8, 2u32,
                                         "Yeah",));
}
