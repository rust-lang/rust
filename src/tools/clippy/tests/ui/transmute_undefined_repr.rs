#![warn(clippy::transmute_undefined_repr)]
#![allow(
    clippy::unit_arg,
    clippy::transmute_ptr_to_ref,
    clippy::useless_transmute,
    clippy::missing_transmute_annotations
)]

use core::any::TypeId;
use core::ffi::c_void;
use core::mem::{MaybeUninit, size_of, transmute};
use core::ptr::NonNull;

fn value<T>() -> T {
    unimplemented!()
}

struct Empty;
struct Ty<T>(T);
struct Ty2<T, U>(T, U);

#[repr(C)]
struct Ty2C<T, U>(T, U);

fn main() {
    unsafe {
        let _: () = transmute(value::<Empty>());
        let _: Empty = transmute(value::<()>());

        let _: Ty<u32> = transmute(value::<u32>());
        let _: Ty<u32> = transmute(value::<u32>());

        // Lint, Ty2 is unordered
        let _: Ty2C<u32, i32> = transmute(value::<Ty2<u32, i32>>());
        //~^ transmute_undefined_repr

        // Lint, Ty2 is unordered
        let _: Ty2<u32, i32> = transmute(value::<Ty2C<u32, i32>>());
        //~^ transmute_undefined_repr

        // Ok, Ty2 types are the same
        let _: Ty2<u32, i32> = transmute(value::<Ty<Ty2<u32, i32>>>());
        // Ok, Ty2 types are the same
        let _: Ty<Ty2<u32, i32>> = transmute(value::<Ty2<u32, i32>>());

        // Lint, different Ty2 instances
        let _: Ty2<u32, f32> = transmute(value::<Ty<Ty2<u32, i32>>>());
        //~^ transmute_undefined_repr

        // Lint, different Ty2 instances
        let _: Ty<Ty2<u32, i32>> = transmute(value::<Ty2<u32, f32>>());
        //~^ transmute_undefined_repr

        let _: Ty<&()> = transmute(value::<&()>());
        let _: &() = transmute(value::<Ty<&()>>());

        // Lint, different Ty2 instances
        let _: &Ty2<u32, f32> = transmute(value::<Ty<&Ty2<u32, i32>>>());
        //~^ transmute_undefined_repr

        // Lint, different Ty2 instances
        let _: Ty<&Ty2<u32, i32>> = transmute(value::<&Ty2<u32, f32>>());
        //~^ transmute_undefined_repr

        // Ok, pointer to usize conversion
        let _: Ty<usize> = transmute(value::<&Ty2<u32, i32>>());
        // Ok, pointer to usize conversion
        let _: &Ty2<u32, i32> = transmute(value::<Ty<usize>>());

        // Ok, transmute to byte array
        let _: Ty<[u8; 8]> = transmute(value::<Ty2<u32, i32>>());
        // Ok, transmute from byte array
        let _: Ty2<u32, i32> = transmute(value::<Ty<[u8; 8]>>());

        // issue #8417
        // Ok, Ty2 types are the same
        let _: Ty2C<Ty2<u32, i32>, ()> = transmute(value::<Ty2<u32, i32>>());
        // Ok, Ty2 types are the same
        let _: Ty2<u32, i32> = transmute(value::<Ty2C<Ty2<u32, i32>, ()>>());

        // Ok, Ty2 types are the same
        let _: &'static mut Ty2<u32, u32> = transmute(value::<Box<Ty2<u32, u32>>>());
        // Ok, Ty2 types are the same
        let _: Box<Ty2<u32, u32>> = transmute(value::<&'static mut Ty2<u32, u32>>());
        // Ok, Ty2 types are the same
        let _: *mut Ty2<u32, u32> = transmute(value::<Box<Ty2<u32, u32>>>());
        // Ok, Ty2 types are the same
        let _: Box<Ty2<u32, u32>> = transmute(value::<*mut Ty2<u32, u32>>());

        // Lint, different Ty2 instances
        let _: &'static mut Ty2<u32, f32> = transmute(value::<Box<Ty2<u32, u32>>>());
        //~^ transmute_undefined_repr

        // Lint, different Ty2 instances
        let _: Box<Ty2<u32, u32>> = transmute(value::<&'static mut Ty2<u32, f32>>());
        //~^ transmute_undefined_repr

        // Ok, type erasure
        let _: *const () = transmute(value::<Ty<&Ty2<u32, f32>>>());
        // Ok, reverse type erasure
        let _: Ty<&Ty2<u32, f32>> = transmute(value::<*const ()>());

        // Ok, type erasure
        let _: *const c_void = transmute(value::<Ty<&Ty2<u32, f32>>>());
        // Ok, reverse type erasure
        let _: Ty<&Ty2<u32, f32>> = transmute(value::<*const c_void>());

        enum Erase {}
        // Ok, type erasure
        let _: *const Erase = transmute(value::<Ty<&Ty2<u32, f32>>>());
        // Ok, reverse type erasure
        let _: Ty<&Ty2<u32, f32>> = transmute(value::<*const Erase>());

        struct Erase2(
            [u8; 0],
            core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
        );
        // Ok, type erasure
        let _: *const Erase2 = transmute(value::<Ty<&Ty2<u32, f32>>>());
        // Ok, reverse type erasure
        let _: Ty<&Ty2<u32, f32>> = transmute(value::<*const Erase2>());

        // Ok, type erasure
        let _: *const () = transmute(value::<&&[u8]>());
        // Ok, reverse type erasure
        let _: &&[u8] = transmute(value::<*const ()>());

        // Ok, type erasure
        let _: *mut c_void = transmute(value::<&mut &[u8]>());
        // Ok, reverse type erasure
        let _: &mut &[u8] = transmute(value::<*mut c_void>());

        // Ok, transmute to byte array
        let _: [u8; size_of::<&[u8]>()] = transmute(value::<&[u8]>());
        // Ok, transmute from byte array
        let _: &[u8] = transmute(value::<[u8; size_of::<&[u8]>()]>());

        // Ok, transmute to int array
        let _: [usize; 2] = transmute(value::<&[u8]>());
        // Ok, transmute from int array
        let _: &[u8] = transmute(value::<[usize; 2]>());

        // Ok
        let _: *const [u8] = transmute(value::<Box<[u8]>>());
        // Ok
        let _: Box<[u8]> = transmute(value::<*mut [u8]>());

        // Ok
        let _: Ty2<u32, u32> = transmute(value::<(Ty2<u32, u32>,)>());
        // Ok
        let _: (Ty2<u32, u32>,) = transmute(value::<Ty2<u32, u32>>());

        // Ok
        let _: Ty2<u32, u32> = transmute(value::<(Ty2<u32, u32>, ())>());
        // Ok
        let _: (Ty2<u32, u32>, ()) = transmute(value::<Ty2<u32, u32>>());

        // Ok
        let _: Ty2<u32, u32> = transmute(value::<((), Ty2<u32, u32>)>());
        // Ok
        let _: ((), Ty2<u32, u32>) = transmute(value::<Ty2<u32, u32>>());

        // Ok
        let _: (usize, usize) = transmute(value::<&[u8]>());
        // Ok
        let _: &[u8] = transmute(value::<(usize, usize)>());

        trait Trait {}
        // Ok
        let _: (isize, isize) = transmute(value::<&dyn Trait>());
        let _: &dyn Trait = transmute(value::<(isize, isize)>());

        // Ok
        let _: MaybeUninit<Ty2<u32, u32>> = transmute(value::<Ty2<u32, u32>>());
        // Ok
        let _: Ty2<u32, u32> = transmute(value::<MaybeUninit<Ty2<u32, u32>>>());

        // Ok
        let _: Ty<&[u32]> = transmute::<&[u32], _>(value::<&Vec<u32>>());

        // Ok
        let _: *const Ty2<u32, u32> = transmute(value::<*const Ty2C<Ty2<u32, u32>, u32>>());
        // Ok
        let _: *const Ty2C<Ty2<u32, u32>, u32> = transmute(value::<*const Ty2<u32, u32>>());
        // Ok
        let _: *const Ty2<u32, u32> = transmute(value::<*const Ty2C<(), Ty2<u32, u32>>>());
        // Ok
        let _: *const Ty2C<(), Ty2<u32, u32>> = transmute(value::<*const Ty2<u32, u32>>());

        // Err
        let _: *const Ty2<u32, u32> = transmute(value::<*const Ty2C<u32, Ty2<u32, u32>>>());
        //~^ transmute_undefined_repr

        // Err
        let _: *const Ty2C<u32, Ty2<u32, u32>> = transmute(value::<*const Ty2<u32, u32>>());
        //~^ transmute_undefined_repr

        // Ok
        let _: NonNull<u8> = transmute(value::<NonNull<(String, String)>>());
        // Ok
        let _: NonNull<(String, String)> = transmute(value::<NonNull<u8>>());
    }
}

fn _with_generics<T: 'static, U: 'static>() {
    if TypeId::of::<T>() != TypeId::of::<u32>() || TypeId::of::<T>() != TypeId::of::<U>() {
        return;
    }
    unsafe {
        // Ok
        let _: &u32 = transmute(value::<&T>());
        // Ok
        let _: &T = transmute(value::<&u32>());

        // Ok
        let _: Vec<U> = transmute(value::<Vec<T>>());
        // Ok
        let _: Vec<T> = transmute(value::<Vec<U>>());

        // Ok
        let _: Ty<&u32> = transmute(value::<&T>());
        // Ok
        let _: Ty<&T> = transmute(value::<&u32>());

        // Ok
        let _: Vec<u32> = transmute(value::<Vec<T>>());
        // Ok
        let _: Vec<T> = transmute(value::<Vec<u32>>());

        // Ok
        let _: &Ty2<u32, u32> = transmute(value::<&Ty2<T, U>>());
        // Ok
        let _: &Ty2<T, U> = transmute(value::<&Ty2<u32, u32>>());

        // Ok
        let _: Vec<Vec<u32>> = transmute(value::<Vec<Vec<T>>>());
        // Ok
        let _: Vec<Vec<T>> = transmute(value::<Vec<Vec<u32>>>());

        // Err
        let _: Vec<Ty2<T, u32>> = transmute(value::<Vec<Ty2<U, i32>>>());
        //~^ transmute_undefined_repr

        // Err
        let _: Vec<Ty2<U, i32>> = transmute(value::<Vec<Ty2<T, u32>>>());
        //~^ transmute_undefined_repr

        // Ok
        let _: *const u32 = transmute(value::<Box<T>>());
        // Ok
        let _: Box<T> = transmute(value::<*const u32>());
    }
}
