#![warn(clippy::uninit_vec)]

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

#[derive(Default)]
struct MyVec {
    vec: Vec<u8>,
}

union MyOwnMaybeUninit {
    value: u8,
    uninit: (),
}

// https://github.com/rust-lang/rust/issues/119620
unsafe fn requires_paramenv<S>() {
    unsafe {
        let mut vec = Vec::<UnsafeCell<*mut S>>::with_capacity(1);
        //~^ uninit_vec
        vec.set_len(1);
    }

    let mut vec = Vec::<UnsafeCell<*mut S>>::with_capacity(2);
    //~^ uninit_vec
    unsafe {
        vec.set_len(2);
    }
}

fn main() {
    // with_capacity() -> set_len() should be detected
    let mut vec: Vec<u8> = Vec::with_capacity(1000);
    //~^ uninit_vec

    unsafe {
        vec.set_len(200);
    }

    // reserve() -> set_len() should be detected
    vec.reserve(1000);
    //~^ uninit_vec

    unsafe {
        vec.set_len(200);
    }

    // new() -> set_len() should be detected
    let mut vec: Vec<u8> = Vec::new();
    //~^ uninit_vec

    unsafe {
        vec.set_len(200);
    }

    // default() -> set_len() should be detected
    let mut vec: Vec<u8> = Default::default();
    //~^ uninit_vec

    unsafe {
        vec.set_len(200);
    }

    let mut vec: Vec<u8> = Vec::default();
    //~^ uninit_vec

    unsafe {
        vec.set_len(200);
    }

    // test when both calls are enclosed in the same unsafe block
    unsafe {
        let mut vec: Vec<u8> = Vec::with_capacity(1000);
        //~^ uninit_vec

        vec.set_len(200);

        vec.reserve(1000);
        //~^ uninit_vec

        vec.set_len(200);
    }

    let mut vec: Vec<u8> = Vec::with_capacity(1000);
    //~^ uninit_vec

    unsafe {
        // test the case where there are other statements in the following unsafe block
        vec.set_len(200);
        assert_eq!(vec.len(), 200);
    }

    // handle vec stored in the field of a struct
    let mut my_vec = MyVec::default();
    my_vec.vec.reserve(1000);
    //~^ uninit_vec

    unsafe {
        my_vec.vec.set_len(200);
    }

    my_vec.vec = Vec::with_capacity(1000);
    //~^ uninit_vec

    unsafe {
        my_vec.vec.set_len(200);
    }

    // Test `#[allow(...)]` attributes on inner unsafe block (shouldn't trigger)
    let mut vec: Vec<u8> = Vec::with_capacity(1000);
    #[allow(clippy::uninit_vec)]
    unsafe {
        vec.set_len(200);
    }

    // MaybeUninit-wrapped types should not be detected
    unsafe {
        let mut vec: Vec<MaybeUninit<u8>> = Vec::with_capacity(1000);
        vec.set_len(200);

        let mut vec: Vec<(MaybeUninit<u8>, MaybeUninit<bool>)> = Vec::with_capacity(1000);
        vec.set_len(200);

        let mut vec: Vec<(MaybeUninit<u8>, [MaybeUninit<bool>; 2])> = Vec::with_capacity(1000);
        vec.set_len(200);
    }

    // known false negative
    let mut vec1: Vec<u8> = Vec::with_capacity(1000);
    let mut vec2: Vec<u8> = Vec::with_capacity(1000);
    unsafe {
        vec1.set_len(200);
        vec2.set_len(200);
    }

    // set_len(0) should not be detected
    let mut vec: Vec<u8> = Vec::with_capacity(1000);
    unsafe {
        vec.set_len(0);
    }

    // ZSTs should not be detected
    let mut vec: Vec<()> = Vec::with_capacity(1000);
    unsafe {
        vec.set_len(10);
    }

    // unions should not be detected
    let mut vec: Vec<MyOwnMaybeUninit> = Vec::with_capacity(1000);
    unsafe {
        vec.set_len(10);
    }

    polymorphic::<()>();

    fn polymorphic<T>() {
        // We are conservative around polymorphic types.
        let mut vec: Vec<T> = Vec::with_capacity(1000);
        //~^ uninit_vec

        unsafe {
            vec.set_len(10);
        }
    }

    fn poly_maybe_uninit<T>() {
        // We are conservative around polymorphic types.
        let mut vec: Vec<MaybeUninit<T>> = Vec::with_capacity(1000);
        unsafe {
            vec.set_len(10);
        }
    }

    fn nested_union<T>() {
        let mut vec: Vec<UnsafeCell<MaybeUninit<T>>> = Vec::with_capacity(1);
        unsafe {
            vec.set_len(1);
        }
    }

    struct Recursive<T>(*const Recursive<T>, MaybeUninit<T>);
    fn recursive_union<T>() {
        // Make sure we don't stack overflow on recursive types.
        // The pointer acts as the base case because it can't be uninit regardless of its pointee.

        let mut vec: Vec<Recursive<T>> = Vec::with_capacity(1);
        //~^ uninit_vec

        unsafe {
            vec.set_len(1);
        }
    }

    #[repr(u8)]
    enum Enum<T> {
        Variant(T),
    }
    fn union_in_enum<T>() {
        // Enums can have a discriminant that can't be uninit, so this should still warn
        let mut vec: Vec<Enum<T>> = Vec::with_capacity(1);
        //~^ uninit_vec

        unsafe {
            vec.set_len(1);
        }
    }
}

mod issue_11715 {
    use std::mem::MaybeUninit;
    #[cfg(target_pointer_width = "64")]
    const HUGE: usize = 4_294_967_296;
    #[cfg(target_pointer_width = "32")]
    const HUGE: usize = 268_435_455;

    fn large_maybeuninit_vec_ice() {
        let mut v: Vec<[MaybeUninit<u64>; HUGE]> = Vec::with_capacity(1);
        unsafe { v.set_len(1) };
    }

    fn large_u8_vec_ice() {
        let mut v: Vec<[u8; HUGE]> = Vec::with_capacity(1);
        //~^ uninit_vec
        unsafe { v.set_len(1) };
    }

    fn large_nested_maybeuninit_vec_ice() {
        let mut v: Vec<[[MaybeUninit<u64>; HUGE]; 2]> = Vec::with_capacity(1);
        unsafe { v.set_len(1) };
    }

    struct HeavyWrapperSafe([MaybeUninit<u64>; HUGE]);
    fn large_struct_maybeuninit_vec_ice() {
        let mut v: Vec<HeavyWrapperSafe> = Vec::with_capacity(1);
        unsafe { v.set_len(1) };
    }

    struct HeavyWrapperUnsafe([u8; HUGE]);
    fn large_struct_u8_vec_ice() {
        let mut v: Vec<HeavyWrapperUnsafe> = Vec::with_capacity(1);
        //~^ uninit_vec
        unsafe { v.set_len(1) };
    }

    #[allow(clippy::large_enum_variant)]
    enum HeavyEnum {
        A([MaybeUninit<u64>; HUGE]),
        B,
    }
    fn large_enum_vec_ice() {
        let mut v: Vec<HeavyEnum> = Vec::with_capacity(1);
        //~^ uninit_vec
        unsafe { v.set_len(1) };
    }

    enum Uninhabited {}
    fn uninhabited_enum() {
        let mut v: Vec<Uninhabited> = Vec::with_capacity(1);
        unsafe { v.set_len(1) };
    }

    enum SingleVariant {
        OnlyOne,
    }
    fn single_variant_enum() {
        let mut v: Vec<SingleVariant> = Vec::with_capacity(1);
        unsafe { v.set_len(1) };
    }

    enum OneVariantU8 {
        ThisOne([u8; HUGE]),
    }
    fn one_variant_u8() {
        let mut v: Vec<OneVariantU8> = Vec::with_capacity(1);
        //~^ uninit_vec
        unsafe { v.set_len(1) };
    }

    enum OneVariantMaybeUninit {
        ThisOne([MaybeUninit<u8>; HUGE]),
    }
    fn one_variant_maybe_uninit() {
        let mut v: Vec<OneVariantMaybeUninit> = Vec::with_capacity(1);
        unsafe { v.set_len(1) };
    }

    fn generic_vec_lints<T>() {
        let mut v: Vec<T> = Vec::with_capacity(1);
        //~^ uninit_vec
        unsafe { v.set_len(1) };
    }

    fn generic_vec_maybeuninit<T>() {
        let mut v: Vec<MaybeUninit<T>> = Vec::with_capacity(1);
        unsafe { v.set_len(1) };
    }

    trait Assoc {
        type Item;
    }
    fn projection_vec_lints<T: Assoc>() {
        let mut v: Vec<<T as Assoc>::Item> = Vec::with_capacity(1);
        //~^ uninit_vec
        unsafe { v.set_len(1) };
    }

    struct Concrete;
    impl Assoc for Concrete {
        type Item = MaybeUninit<u8>;
    }
    fn normalized_projection_vec_ok() {
        let mut v: Vec<<Concrete as Assoc>::Item> = Vec::with_capacity(1);
        unsafe { v.set_len(1) };
    }

    enum E<T, U> {
        Foo(MaybeUninit<T>),
        Bar(U),
    }
    fn enum_uninhabited_zst() {
        let mut v: Vec<E<u8, core::convert::Infallible>> = Vec::with_capacity(1);
        unsafe { v.set_len(1) };
    }
    fn enum_uninhabited_non_zst() {
        let mut v: Vec<E<u8, (u8, core::convert::Infallible)>> = Vec::with_capacity(1);
        //~^ uninit_vec
        unsafe { v.set_len(1) };
    }
}
