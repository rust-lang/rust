/* Module: math */

#[link_name = ""]
#[abi = "cdecl"]
native mod libc {
    fn sqrt(n: float) -> float;
    fn sin(n: float) -> float;
    fn asin(n: float) -> float;
    fn cos(n: float) -> float;
    fn acos(n: float) -> float;
    fn tan(n: float) -> float;
    fn atan(n: float) -> float;
}

/*
Function: sqrt

Returns the square root
*/
fn sqrt(x: float) -> float { libc::sqrt(x) }

/*
Function: sin

Returns the sine of an angle
*/
fn sin(x: float) -> float { libc::sin(x) }

/*
Function: cos

Returns the cosine of an angle
*/
fn cos(x: float) -> float { libc::cos(x) }

/*
Function: tan

Returns the tangent of an angle
*/
fn tan(x: float) -> float { libc::tan(x) }

/*
Function: asin

Returns the arcsine of an angle
*/
fn asin(x: float) -> float { libc::asin(x) }

/*
Function: acos

Returns the arccosine of an angle
*/
fn acos(x: float) -> float { libc::acos(x) }

/*
Function: atan

Returns the arctangent of an angle
*/
fn atan(x: float) -> float { libc::atan(x) }

/*
Const: pi

Archimedes' constant
*/
const pi: float = 3.141592653589793;

/*
Function: min

Returns the minimum of two values
*/
fn min<T>(x: T, y: T) -> T { x < y ? x : y }

/*
Function: max

Returns the maximum of two values
*/
fn max<T>(x: T, y: T) -> T { x < y ? y : x }
