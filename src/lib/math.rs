/* Module: math */

native "llvm" mod llvm {
    fn sqrt(n: float) -> float = "sqrt.f64";
    fn sin(n: float) -> float = "sin.f64";
    fn asin(n: float) -> float = "asin.f64";
    fn cos(n: float) -> float = "cos.f64";
    fn acos(n: float) -> float = "acos.f64";
    fn tan(n: float) -> float = "tan.f64";
    fn atan(n: float) -> float = "atan.f64";
}

/*
Function: sqrt

Returns the square root
*/
fn sqrt(x: float) -> float { llvm::sqrt(x) }

/*
Function: sin

Returns the sine of an angle
*/
fn sin(x: float) -> float { llvm::sin(x) }

/*
Function: cos

Returns the cosine of an angle
*/
fn cos(x: float) -> float { llvm::cos(x) }

/*
Function: tan

Returns the tangent of an angle
*/
fn tan(x: float) -> float { llvm::tan(x) }

/*
Function: asin

Returns the arcsine of an angle
*/
fn asin(x: float) -> float { llvm::asin(x) }

/*
Function: acos

Returns the arccosine of an angle
*/
fn acos(x: float) -> float { llvm::acos(x) }

/*
Function: atan

Returns the arctangent of an angle
*/
fn atan(x: float) -> float { llvm::atan(x) }

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
