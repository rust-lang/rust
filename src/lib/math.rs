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
    #[link_name="log"]
    fn ln(n: float) -> float;
    fn log2(n: float) -> float;
    fn log10(n: float) -> float;
    fn log1p(n: float) -> float;
}

/*
Function: sqrt

Returns the square root
*/
pure fn sqrt(x: float) -> float { unsafe { libc::sqrt(x) } }

/*
Function: sin

Returns the sine of an angle
*/
pure fn sin(x: float) -> float { unsafe { libc::sin(x) } }

/*
Function: cos

Returns the cosine of an angle
*/
pure fn cos(x: float) -> float { unsafe { libc::cos(x) } }

/*
Function: tan

Returns the tangent of an angle
*/
pure fn tan(x: float) -> float { unsafe { libc::tan(x) } }

/*
Function: asin

Returns the arcsine of an angle
*/
pure fn asin(x: float) -> float { unsafe { libc::asin(x) } }

/*
Function: acos

Returns the arccosine of an angle
*/
pure fn acos(x: float) -> float { unsafe { libc::acos(x) } }

/*
Function: atan

Returns the arctangent of an angle
*/
pure fn atan(x: float) -> float { unsafe { libc::atan(x) } }

/*
Const: pi

Archimedes' constant
*/
const pi: float = 3.141592653589793;

/*
Function: min

Returns the minimum of two values
*/
pure fn min<copy T>(x: T, y: T) -> T { x < y ? x : y }

/*
Function: max

Returns the maximum of two values
*/
pure fn max<copy T>(x: T, y: T) -> T { x < y ? y : x }

/*
Const: e

Euler's number
*/
const e: float = 2.718281828459045235;

/*
Function: ln

Returns the natural logaritm
*/
pure fn ln(n: float) -> float { unsafe { libc::ln(n) } }

/*
Function: log2

Returns the logarithm to base 2
*/
pure fn log2(n: float) -> float { unsafe { libc::log2(n) } }

/*
Function: log2

Returns the logarithm to base 10
*/
pure fn log10(n: float) -> float { unsafe { libc::log10(n) } }


/*
Function: log1p

Returns the natural logarithm of `1+n` accurately,
even for very small values of `n`
*/
pure fn ln1p(n: float) -> float { unsafe { libc::log1p(n) } }
