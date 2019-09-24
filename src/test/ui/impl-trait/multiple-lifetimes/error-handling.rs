// compile-flags:-Zborrowck=mir

#![feature(member_constraints)]
#![feature(type_alias_impl_trait)]

#[derive(Clone)]
struct CopyIfEq<T, U>(T, U);

impl<T: Copy> Copy for CopyIfEq<T, T> {}

type E<'a, 'b> = impl Sized;

fn foo<'a, 'b, 'c>(x: &'static i32, mut y: &'a i32) -> E<'b, 'c> {
    //~^ ERROR lifetime may not live long enough
    let v = CopyIfEq::<*mut _, *mut _>(&mut {x}, &mut y);
    let u = v;
    let _: *mut &'a i32 = u.1;
    unsafe { let _: &'b i32 = *u.0; }
    u.0
}

fn main() {}
