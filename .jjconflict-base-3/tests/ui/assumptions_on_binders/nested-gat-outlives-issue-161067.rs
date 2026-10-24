//@ compile-flags: -Zassumptions-on-binders
//@ needs-rustc-debug-assertions

// Regression test for #161067. A nested non-rigid alias must be normalized
// before it reaches lexical region solving through `ty_known_to_outlive`.

struct D;

trait Des {
    type Out<'x, T>;
    //~^ ERROR missing required bound on `Out`

    fn des<'z>() -> Self::Out<'z, Self::Out<'z, D>>;
}

fn main() {}
