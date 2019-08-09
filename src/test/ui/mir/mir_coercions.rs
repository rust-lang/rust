// run-pass
#![feature(coerce_unsized, unsize)]

use std::ops::CoerceUnsized;
use std::marker::Unsize;

fn identity_coercion(x: &(dyn Fn(u32)->u32 + Send)) -> &dyn Fn(u32)->u32 {
    x
}
fn fn_coercions(f: &fn(u32) -> u32) ->
    (unsafe fn(u32) -> u32,
     &(dyn Fn(u32) -> u32+Send))
{
    (*f, f)
}

fn simple_array_coercion(x: &[u8; 3]) -> &[u8] { x }

fn square(a: u32) -> u32 { a * a }

#[derive(PartialEq,Eq)]
struct PtrWrapper<'a, T: 'a+?Sized>(u32, u32, (), &'a T);
impl<'a, T: ?Sized+Unsize<U>, U: ?Sized>
    CoerceUnsized<PtrWrapper<'a, U>> for PtrWrapper<'a, T> {}

struct TrivPtrWrapper<'a, T: 'a+?Sized>(&'a T);
impl<'a, T: ?Sized+Unsize<U>, U: ?Sized>
    CoerceUnsized<TrivPtrWrapper<'a, U>> for TrivPtrWrapper<'a, T> {}

fn coerce_ptr_wrapper(p: PtrWrapper<[u8; 3]>) -> PtrWrapper<[u8]> {
    p
}

fn coerce_triv_ptr_wrapper(p: TrivPtrWrapper<[u8; 3]>) -> TrivPtrWrapper<[u8]> {
    p
}

fn coerce_fat_ptr_wrapper(p: PtrWrapper<dyn Fn(u32) -> u32+Send>)
                          -> PtrWrapper<dyn Fn(u32) -> u32> {
    p
}

fn coerce_ptr_wrapper_poly<'a, T, Trait: ?Sized>(p: PtrWrapper<'a, T>)
                                                 -> PtrWrapper<'a, Trait>
    where PtrWrapper<'a, T>: CoerceUnsized<PtrWrapper<'a, Trait>>
{
    p
}

fn main() {
    let a = [0,1,2];
    let square_local : fn(u32) -> u32 = square;
    let (f,g) = fn_coercions(&square_local);
    assert_eq!(f as usize, square as usize);
    assert_eq!(g(4), 16);
    assert_eq!(identity_coercion(g)(5), 25);

    assert_eq!(simple_array_coercion(&a), &a);
    let w = coerce_ptr_wrapper(PtrWrapper(2,3,(),&a));
    assert!(w == PtrWrapper(2,3,(),&a) as PtrWrapper<[u8]>);

    let w = coerce_triv_ptr_wrapper(TrivPtrWrapper(&a));
    assert_eq!(&w.0, &a);

    let z = coerce_fat_ptr_wrapper(PtrWrapper(2,3,(),&square_local));
    assert_eq!((z.3)(6), 36);

    let z: PtrWrapper<dyn Fn(u32) -> u32> =
        coerce_ptr_wrapper_poly(PtrWrapper(2,3,(),&square_local));
    assert_eq!((z.3)(6), 36);
}
