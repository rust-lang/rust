//@ignore-bitwidth: 32

#![warn(clippy::result_large_err)]
#![allow(clippy::large_enum_variant)]

pub fn small_err() -> Result<(), u128> {
    Ok(())
}

pub fn large_err() -> Result<(), [u8; 512]> {
    //~^ result_large_err

    Ok(())
}

pub struct FullyDefinedLargeError {
    _foo: u128,
    _bar: [u8; 100],
    _foobar: [u8; 120],
}

impl FullyDefinedLargeError {
    pub fn ret() -> Result<(), Self> {
        //~^ result_large_err

        Ok(())
    }
}

pub fn struct_error() -> Result<(), FullyDefinedLargeError> {
    //~^ result_large_err

    Ok(())
}

type Fdlr<T> = std::result::Result<T, FullyDefinedLargeError>;
pub fn large_err_via_type_alias<T>(x: T) -> Fdlr<T> {
    //~^ result_large_err

    Ok(x)
}

pub fn param_small_error<R>() -> Result<(), (R, u128)> {
    Ok(())
}

pub fn param_large_error<R>() -> Result<(), (u128, R, FullyDefinedLargeError)> {
    //~^ result_large_err

    Ok(())
}

pub enum LargeErrorVariants<T> {
    _Small(u8),
    _Omg([u8; 512]),
    _Param(T),
}

impl LargeErrorVariants<()> {
    pub fn large_enum_error() -> Result<(), Self> {
        //~^ result_large_err

        Ok(())
    }
}

enum MultipleLargeVariants {
    _Biggest([u8; 1024]),
    _AlsoBig([u8; 512]),
    _Ok(usize),
}

impl MultipleLargeVariants {
    fn large_enum_error() -> Result<(), Self> {
        //~^ result_large_err

        Ok(())
    }
}

trait TraitForcesLargeError {
    fn large_error() -> Result<(), [u8; 512]> {
        //~^ result_large_err

        Ok(())
    }
}

struct TraitImpl;

impl TraitForcesLargeError for TraitImpl {
    // Should not lint
    fn large_error() -> Result<(), [u8; 512]> {
        Ok(())
    }
}

pub union FullyDefinedUnionError {
    _maybe: u8,
    _or_even: [[u8; 16]; 32],
}

pub fn large_union_err() -> Result<(), FullyDefinedUnionError> {
    //~^ result_large_err

    Ok(())
}

pub union UnionError<T: Copy> {
    _maybe: T,
    _or_perhaps_even: (T, [u8; 512]),
}

pub fn param_large_union<T: Copy>() -> Result<(), UnionError<T>> {
    //~^ result_large_err

    Ok(())
}

pub struct ArrayError<T, U> {
    _large_array: [T; 32],
    _other_stuff: U,
}

pub fn array_error_subst<U>() -> Result<(), ArrayError<i32, U>> {
    //~^ result_large_err

    Ok(())
}

pub fn array_error<T, U>() -> Result<(), ArrayError<(i32, T), U>> {
    //~^ result_large_err

    Ok(())
}

// Issue #10005
enum Empty {}
fn _empty_error() -> Result<(), Empty> {
    Ok(())
}

fn main() {}
