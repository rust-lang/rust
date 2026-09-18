struct S<const N: usize>;
impl<const N: usize> S<N> {
    const LEN: usize = core::direct_const_arg!(1);
    //~^ ERROR: use of unstable library feature `min_generic_const_args` [E0658]
    //~| ERROR: expected expression, found `direct_const_arg!()`
    fn arr() {
        [8; Self::LEN]
    }
}

pub fn main() {}
