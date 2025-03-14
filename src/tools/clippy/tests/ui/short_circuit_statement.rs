#![warn(clippy::short_circuit_statement)]
#![allow(clippy::nonminimal_bool)]

fn main() {
    f() && g();
    //~^ short_circuit_statement

    f() || g();
    //~^ short_circuit_statement

    1 == 2 || g();
    //~^ short_circuit_statement

    (f() || g()) && (H * 2);
    //~^ short_circuit_statement

    (f() || g()) || (H * 2);
    //~^ short_circuit_statement

    macro_rules! mac {
        ($f:ident or $g:ident) => {
            $f() || $g()
        };
        ($f:ident and $g:ident) => {
            $f() && $g()
        };
        () => {
            f() && g()
        };
    }

    mac!() && mac!();
    //~^ short_circuit_statement

    mac!() || mac!();
    //~^ short_circuit_statement

    // Do not lint if the expression comes from a macro
    mac!();
}

fn f() -> bool {
    true
}

fn g() -> bool {
    false
}

struct H;

impl std::ops::Mul<u32> for H {
    type Output = bool;
    fn mul(self, other: u32) -> Self::Output {
        true
    }
}
