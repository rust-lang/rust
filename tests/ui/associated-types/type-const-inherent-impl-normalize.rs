struct S<const N: usize>;
impl<const N: usize> S<N> {
    const LEN: usize = std::gca!(1);
    //~^ ERROR: use of unstable library feature `min_generic_const_args` [E0658]
    //~| ERROR: expected expression, found `gca!()`
    fn arr() {
        [8; Self::LEN]
    }
}

pub fn main() {}
