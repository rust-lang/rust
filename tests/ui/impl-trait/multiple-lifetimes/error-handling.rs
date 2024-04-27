#![feature(type_alias_impl_trait)]

#[derive(Clone)]
struct CopyIfEq<T, U>(T, U);

impl<T: Copy> Copy for CopyIfEq<T, T> {}

type E<'a, 'b> = impl Sized;

fn foo<'a, 'b, 'c>(x: &'static i32, mut y: &'a i32) -> E<'b, 'c> {
    let v = CopyIfEq::<*mut _, *mut _>(&mut { x }, &mut y);

    // This assignment requires that `x` and `y` have the same type due to the
    // `Copy` impl. The reason why we are using a copy to create a constraint
    // is that only borrow checking (not regionck in type checking) enforces
    // this bound.
    let u = v;
    let _: *mut &'a i32 = u.1;
    unsafe {
        let _: &'b i32 = *u.0;
        //~^ ERROR lifetime may not live long enough
    }
    u.0
}

fn main() {}
