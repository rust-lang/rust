#![allow(improper_ctypes_definitions)]

use std::fmt;

trait FalsePositive {
    fn to_string(&self) -> String;
}

struct A;
struct B;
struct C;
struct D;
struct E;
struct F;
struct G;
struct H;
struct I;
struct J;

impl A {
    // Should be detected; emit warning
    fn to_string(&self) -> String {
        //~^ ERROR: implementation of inherent method `to_string(&self) -> String` for type `A
        "A.to_string()".to_string()
    }

    // Should not be detected as it does not match the function signature
    fn to_str(&self) -> String {
        "A.to_str()".to_string()
    }
}

// Should not be detected as it is a free function
fn to_string() -> String {
    "free to_string()".to_string()
}

impl B {
    // Should not be detected, wrong return type
    fn to_string(&self) -> i32 {
        42
    }
}

impl C {
    // Should be detected and emit error as C also implements Display
    fn to_string(&self) -> String {
        //~^ ERROR: type `C` implements inherent method `to_string(&self) -> String` which sha
        "C.to_string()".to_string()
    }
}

impl fmt::Display for C {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "impl Display for C")
    }
}

impl FalsePositive for D {
    // Should not be detected, as it is a trait function
    fn to_string(&self) -> String {
        "impl FalsePositive for D".to_string()
    }
}

impl E {
    // Should not be detected, as it is not bound to an instance
    fn to_string() -> String {
        "E::to_string()".to_string()
    }
}

impl F {
    // Should not be detected, as it does not match the function signature
    fn to_string(&self, _i: i32) -> String {
        "F.to_string()".to_string()
    }
}

impl G {
    // Should not be detected, as it does not match the function signature
    fn to_string<const _N: usize>(&self) -> String {
        "G.to_string()".to_string()
    }
}

// Issue #11201

impl H {
    unsafe fn to_string(&self) -> String {
        "G.to_string()".to_string()
    }
}

impl I {
    extern "C" fn to_string(&self) -> String {
        "G.to_string()".to_string()
    }
}

impl J {
    unsafe extern "C" fn to_string(&self) -> String {
        "G.to_string()".to_string()
    }
}

fn main() {
    let a = A;
    a.to_string();
    a.to_str();

    to_string();

    let b = B;
    b.to_string();

    let c = C;
    C.to_string();

    let d = D;
    d.to_string();

    E::to_string();

    let f = F;
    f.to_string(1);

    let g = G;
    g.to_string::<1>();
}
