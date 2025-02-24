#![warn(clippy::short_circuit_statement)]
#![allow(clippy::nonminimal_bool)]

fn main() {
    f() && g();
    //~^ ERROR: boolean short circuit operator in statement
    f() || g();
    //~^ ERROR: boolean short circuit operator in statement
    1 == 2 || g();
    //~^ ERROR: boolean short circuit operator in statement
    (f() || g()) && (H * 2);
    //~^ ERROR: boolean short circuit operator in statement
    (f() || g()) || (H * 2);
    //~^ ERROR: boolean short circuit operator in statement

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
    //~^ ERROR: boolean short circuit operator in statement
    mac!() || mac!();
    //~^ ERROR: boolean short circuit operator in statement

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
