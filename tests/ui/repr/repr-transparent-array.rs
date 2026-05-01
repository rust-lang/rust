#![deny(repr_transparent_non_zst_fields)]

type SusArray = [u8; 0];

type GoodArray = [[(); 0]; 42];

pub type Sized = i32;

#[repr(Rust, packed(1))]
pub struct RustPacked1Arr<T>([*const T; 0]);

#[repr(C, packed(1))]
pub struct CPacked1Arr<T>([*const T; 0]);

#[repr(Rust, packed(2))]
pub struct RustPacked2Arr([u8; 0]);

#[repr(transparent)]
pub struct T1(SusArray);
#[repr(transparent)]
pub struct T2(GoodArray, SusArray);
#[repr(transparent)]
pub struct T3(SusArray, GoodArray);
#[repr(transparent)]
pub struct T4(Sized, RustPacked1Arr<u128>);

#[repr(transparent)]
pub struct T5(Sized, SusArray);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain array types with non-trivial element types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T6(SusArray, Sized);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain array types with non-trivial element types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T7(T1, SusArray); // still wrong, even when the array is hidden inside another type
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain array types with non-trivial element types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T8(SusArray, SusArray); // still wrong
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain array types with non-trivial element types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T9([[u8; 0]; 0], SusArray); // still wrong
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain array types with non-trivial element types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T10(Sized, CPacked1Arr<u128>);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain `repr(C)` types
//~| WARN this was previously accepted by the compiler

#[repr(transparent)]
pub struct T11(Sized, RustPacked2Arr);
//~^ ERROR zero-sized fields in `repr(transparent)` cannot contain array types with non-trivial element types
//~| WARN this was previously accepted by the compiler

fn main() {}
