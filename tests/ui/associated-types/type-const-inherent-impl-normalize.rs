struct S<const N: usize>;
impl<const N: usize> S<N> {
    type const LEN: usize = 1;
    //~^ ERROR: associated `type const` are unstable [E0658]
    //~| ERROR: `type const` syntax is experimental [E0658]
    fn arr() {
        [8; Self::LEN]
        //~^ WARN: cannot use constants which depend on generic parameters in types
        //~| WARN: this was previously accepted by the compiler but is being phased out
        //~| WARN: cannot use constants which depend on generic parameters in types
        //~| WARN: this was previously accepted by the compiler but is being phased out
        //~| ERROR: mismatched types
    }
}

pub fn main() {}
