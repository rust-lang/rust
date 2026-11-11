// Regression test for #132955 checking that we handle anon consts with
// inference variables in their generic arguments correctly.
//
// This arose via diagnostics where we would have some failing goal such
// as `[u8; AnonConst<Self>]: PartialEq<Self::A>`, then as part of diagnostics
// we would replace all generic parameters with inference vars which would yield
// a self type of `[u8; AnonConst<?x>]` and then attempt to normalize `AnonConst<?x>`.

pub trait T {
    type A;
    const P: Self::A;

    fn a() {
        [0u8; std::mem::size_of::<Self::A>()] == Self::P;
        //~^ ERROR: can't compare
        //~| ERROR: constant expression depends on a generic parameter
        //~| ERROR: constant expression depends on a generic parameter
    }
}

fn main() {}
