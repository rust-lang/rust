//! Test that `Copy` cannot be implemented if any field doesn't implement `Copy`.

struct CantCopyThis;

struct IWantToCopyThis {
    but_i_cant: CantCopyThis,
}

impl Copy for IWantToCopyThis {}
//~^ ERROR the trait `Copy` cannot be implemented for this type

enum CantCopyThisEither {
    A,
    B,
}

enum IWantToCopyThisToo {
    ButICant(CantCopyThisEither),
}

impl Copy for IWantToCopyThisToo {}
//~^ ERROR the trait `Copy` cannot be implemented for this type

fn main() {}
