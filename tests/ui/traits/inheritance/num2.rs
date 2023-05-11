// run-pass
// A more complex example of numeric extensions

pub trait TypeExt {}

impl TypeExt for u8 {}
impl TypeExt for u16 {}
impl TypeExt for u32 {}
impl TypeExt for u64 {}
impl TypeExt for usize {}

impl TypeExt for i8 {}
impl TypeExt for i16 {}
impl TypeExt for i32 {}
impl TypeExt for i64 {}
impl TypeExt for isize {}

impl TypeExt for f32 {}
impl TypeExt for f64 {}


pub trait NumExt: TypeExt + PartialEq + PartialOrd {}

impl NumExt for u8 {}
impl NumExt for u16 {}
impl NumExt for u32 {}
impl NumExt for u64 {}
impl NumExt for usize {}

impl NumExt for i8 {}
impl NumExt for i16 {}
impl NumExt for i32 {}
impl NumExt for i64 {}
impl NumExt for isize {}

impl NumExt for f32 {}
impl NumExt for f64 {}


pub trait UnSignedExt: NumExt {}

impl UnSignedExt for u8 {}
impl UnSignedExt for u16 {}
impl UnSignedExt for u32 {}
impl UnSignedExt for u64 {}
impl UnSignedExt for usize {}


pub trait SignedExt: NumExt {}

impl SignedExt for i8 {}
impl SignedExt for i16 {}
impl SignedExt for i32 {}
impl SignedExt for i64 {}
impl SignedExt for isize {}

impl SignedExt for f32 {}
impl SignedExt for f64 {}


pub trait IntegerExt: NumExt {}

impl IntegerExt for u8 {}
impl IntegerExt for u16 {}
impl IntegerExt for u32 {}
impl IntegerExt for u64 {}
impl IntegerExt for usize {}

impl IntegerExt for i8 {}
impl IntegerExt for i16 {}
impl IntegerExt for i32 {}
impl IntegerExt for i64 {}
impl IntegerExt for isize {}


pub trait FloatExt: NumExt {}

impl FloatExt for f32 {}
impl FloatExt for f64 {}


fn test_float_ext<T:FloatExt>(n: T) { println!("{}", n < n) }

pub fn main() {
    test_float_ext(1f32);
}
