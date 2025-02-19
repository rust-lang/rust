//@ proc-macro: malicious-macro.rs
#![feature(derive_coerce_pointee, arbitrary_self_types)]

extern crate core;
extern crate malicious_macro;

use std::marker::CoercePointee;

#[derive(CoercePointee)]
//~^ ERROR: `CoercePointee` can only be derived on `struct`s with `#[repr(transparent)]`
//~| NOTE: in this expansion of #[derive(CoercePointee)]
enum NotStruct<'a, T: ?Sized> {
    Variant(&'a T),
}

#[derive(CoercePointee)]
//~^ ERROR: `CoercePointee` can only be derived on `struct`s with at least one field
//~| NOTE: in this expansion of #[derive(CoercePointee)]
#[repr(transparent)]
struct NoField<#[pointee] T: ?Sized> {}
//~^ ERROR: type parameter `T` is never used
//~| NOTE: unused type parameter
//~| HELP: consider removing `T`, referring to it in a field, or using a marker such as `PhantomData`

#[derive(CoercePointee)]
//~^ ERROR: `CoercePointee` can only be derived on `struct`s with at least one field
//~| NOTE: in this expansion of #[derive(CoercePointee)]
#[repr(transparent)]
struct NoFieldUnit<#[pointee] T: ?Sized>();
//~^ ERROR: type parameter `T` is never used
//~| NOTE: unused type parameter
//~| HELP: consider removing `T`, referring to it in a field, or using a marker such as `PhantomData`

#[derive(CoercePointee)]
//~^ ERROR: `CoercePointee` can only be derived on `struct`s that are generic over at least one type
//~| NOTE: in this expansion of #[derive(CoercePointee)]
#[repr(transparent)]
struct NoGeneric<'a>(&'a u8);

#[derive(CoercePointee)]
//~^ ERROR: exactly one generic type parameter must be marked as `#[pointee]` to derive `CoercePointee` traits
//~| NOTE: in this expansion of #[derive(CoercePointee)]
#[repr(transparent)]
struct AmbiguousPointee<'a, T1: ?Sized, T2: ?Sized> {
    a: (&'a T1, &'a T2),
}

#[derive(CoercePointee)]
#[repr(transparent)]
struct TooManyPointees<'a, #[pointee] A: ?Sized, #[pointee] B: ?Sized>((&'a A, &'a B));
//~^ ERROR: only one type parameter can be marked as `#[pointee]` when deriving `CoercePointee` traits
//~| NOTE: here another type parameter is marked as `#[pointee]`

#[derive(CoercePointee)]
struct NotTransparent<'a, #[pointee] T: ?Sized> {
    //~^ ERROR: `derive(CoercePointee)` is only applicable to `struct` with `repr(transparent)` layout
    ptr: &'a T,
}

#[derive(CoercePointee)]
#[repr(transparent)]
struct NoMaybeSized<'a, #[pointee] T> {
    //~^ ERROR: `derive(CoercePointee)` requires `T` to be marked `?Sized`
    ptr: &'a T,
}

#[derive(CoercePointee)]
#[repr(transparent)]
struct PointeeOnField<'a, #[pointee] T: ?Sized> {
    #[pointee]
    //~^ ERROR: the `#[pointee]` attribute may only be used on generic parameters
    ptr: &'a T,
}

#[derive(CoercePointee)]
#[repr(transparent)]
struct PointeeInTypeConstBlock<
    'a,
    T: ?Sized = [u32; const {
                    struct UhOh<#[pointee] T>(T);
                    //~^ ERROR: the `#[pointee]` attribute may only be used on generic parameters
                    10
                }],
> {
    ptr: &'a T,
}

#[derive(CoercePointee)]
#[repr(transparent)]
struct PointeeInConstConstBlock<
    'a,
    T: ?Sized,
    const V: u32 = {
        struct UhOh<#[pointee] T>(T);
        //~^ ERROR: the `#[pointee]` attribute may only be used on generic parameters
        10
    },
> {
    ptr: &'a T,
}

#[derive(CoercePointee)]
#[repr(transparent)]
struct PointeeInAnotherTypeConstBlock<'a, #[pointee] T: ?Sized> {
    ptr: PointeeInConstConstBlock<
        'a,
        T,
        {
            struct UhOh<#[pointee] T>(T);
            //~^ ERROR: the `#[pointee]` attribute may only be used on generic parameters
            0
        },
    >,
}

// However, reordering attributes should work nevertheless.
#[repr(transparent)]
#[derive(CoercePointee)]
struct ThisIsAPossibleCoercePointee<'a, #[pointee] T: ?Sized> {
    ptr: &'a T,
}

// Also, these paths to Sized should work
#[derive(CoercePointee)]
#[repr(transparent)]
struct StdSized<'a, #[pointee] T: ?std::marker::Sized> {
    ptr: &'a T,
}
#[derive(CoercePointee)]
#[repr(transparent)]
struct CoreSized<'a, #[pointee] T: ?core::marker::Sized> {
    ptr: &'a T,
}
#[derive(CoercePointee)]
#[repr(transparent)]
struct GlobalStdSized<'a, #[pointee] T: ?::std::marker::Sized> {
    ptr: &'a T,
}
#[derive(CoercePointee)]
#[repr(transparent)]
struct GlobalCoreSized<'a, #[pointee] T: ?::core::marker::Sized> {
    ptr: &'a T,
}

#[derive(CoercePointee)]
#[malicious_macro::norepr]
#[repr(transparent)]
struct TryToWipeRepr<'a, #[pointee] T: ?Sized> {
    //~^ ERROR: `derive(CoercePointee)` is only applicable to `struct` with `repr(transparent)` layout [E0802]
    ptr: &'a T,
}

#[repr(transparent)]
#[derive(CoercePointee)]
struct RcWithId<T: ?Sized> {
    inner: std::rc::Rc<(i32, Box<T>)>,
    //~^ ERROR: `Box<T>` cannot be coerced to an unsized value [E0802]
    //~| NOTE: `derive(CoercePointee)` demands that `Box<T>` can be coerced to an unsized type
    //~| HELP: `derive(CoercePointee)` requires exactly one copy of `#[pointee]` type at the end of the `struct` definition, without any further pointer or reference indirection
    //~| ERROR: `Box<T>` cannot be coerced to an unsized value [E0802]
    //~| NOTE: `derive(CoercePointee)` demands that `Box<T>` can be coerced to an unsized type
    //~| HELP: `derive(CoercePointee)` requires exactly one copy of `#[pointee]` type at the end of the `struct` definition, without any further pointer or reference indirection
    //~| NOTE: duplicate diagnostic emitted due to `-Z deduplicate-diagnostics=no`
}

#[repr(transparent)]
#[derive(CoercePointee)]
struct MoreThanOneField<T: ?Sized> {
    //~^ ERROR: transparent struct needs at most one field with non-trivial size or alignment, but has 2 [E0690]
    //~| NOTE: needs at most one field with non-trivial size or alignment, but has 2
    inner1: Box<T>,
    //~^ ERROR: `derive(CoercePointee)` only admits exactly one data field, to which `dyn` methods shall be dispatched [E0802]
    //~| ERROR: `derive(CoercePointee)` only admits exactly one data field, on which unsize coercion shall be performed [E0802]
    //~| NOTE: this field has non-zero size or requires alignment
    inner2: Box<T>,
    //~^ NOTE: this field has non-zero size or requires alignment
}

struct NotCoercePointeeData<T: ?Sized>(T);

#[repr(transparent)]
#[derive(CoercePointee)]
struct UsingNonCoercePointeeData<T: ?Sized>(NotCoercePointeeData<T>);
//~^ ERROR: `NotCoercePointeeData<T>` cannot be coerced to an unsized type [E0802]
//~| NOTE: `derive(CoercePointee)` demands that `NotCoercePointeeData<T>` can be coerced to an unsized type
//~| HELP: the standard pointers such as `Arc`, `Rc`, `Box`, and other types with `derive(CoercePointee)` can be coerced to their corresponding unsized types
//~| ERROR: `NotCoercePointeeData<T>` cannot be coerced to an unsized type, to which `dyn` methods can be dispatched [E0802]
//~| NOTE: `derive(CoercePointee)` demands that `dyn` methods can be dispatched when `NotCoercePointeeData<T>` can be coerced to an unsized type
//~| HELP: `dyn` methods can be dispatched to the standard pointers such as `Arc`, `Rc`, `Box`, and other types with `derive(CoercePointee)`


fn main() {}
