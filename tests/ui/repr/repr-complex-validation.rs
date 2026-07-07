// Check that `#[repr(complex)]` validates the shape of the type it is applied to:
// it must be a struct with exactly two fields of the same type.
#![feature(repr_complex)]
#![crate_type = "lib"]

#[repr(complex)]
struct Zero; //~ ERROR `repr(complex)` type must have two fields

#[repr(complex)]
struct One(f32); //~ ERROR `repr(complex)` type must have two fields

#[repr(complex)]
struct Three(f32, f32, f32); //~ ERROR `repr(complex)` type cannot have more than two fields

#[repr(complex)]
struct DifferentTypes(f32, f64); //~ ERROR `repr(complex)` type must have two fields of the same type

// These are well-formed.
#[repr(complex)]
struct GoodTuple(f32, f32);

#[repr(complex)]
struct GoodNamed {
    re: f64,
    im: f64,
}

#[repr(complex)]
struct GoodGeneric<T>(T, T);
