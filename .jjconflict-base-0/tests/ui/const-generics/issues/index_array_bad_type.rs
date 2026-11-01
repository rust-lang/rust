struct Struct<const N: i128>(pub [u8; N]);
//~^ ERROR the constant `N` is not of type `usize`

pub fn function(value: Struct<3>) -> u8 {
    value.0[0]
    //~^ ERROR the constant `3` is not of type `usize`

    // FIXME(const_generics): Ideally we wouldn't report the above error
    // b/c `Struct<_>` is never well formed, but I'd rather report too many
    // errors rather than ICE the compiler.
}

fn main() {}
