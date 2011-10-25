native "llvm" mod llvm {
    fn sqrt(n: float) -> float = "sqrt.f64";
    fn sin(n: float) -> float = "sin.f64";
    fn asin(n: float) -> float = "asin.f64";
    fn cos(n: float) -> float = "cos.f64";
    fn acos(n: float) -> float = "acos.f64";
    fn tan(n: float) -> float = "tan.f64";
    fn atan(n: float) -> float = "atan.f64";
}

fn sqrt(x: float) -> float { llvm::sqrt(x) }
fn sin(x: float) -> float { llvm::sin(x) }
fn cos(x: float) -> float { llvm::cos(x) }
fn tan(x: float) -> float { llvm::tan(x) }
fn asin(x: float) -> float { llvm::asin(x) }
fn acos(x: float) -> float { llvm::acos(x) }
fn atan(x: float) -> float { llvm::atan(x) }

const pi: float = 3.141592653589793;

fn min<T>(x: T, y: T) -> T { x < y ? x : y }
fn max<T>(x: T, y: T) -> T { x < y ? y : x }
