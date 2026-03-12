//@ known-bug: #146965
//@ compile-flags: --crate-type lib -C opt-level=3

fn conjure<T>() -> T {
    panic!()
}

pub trait Ring {
    type Element;
}
impl<T> Ring for T {
    type Element = u16;
}

// Removing the : Ring bound makes it not ICE
fn map_coeff<T: Ring>(f: impl Fn(<T as Ring>::Element)) {
    let c = conjure::<<T as Ring>::Element>();
    f(c);
}

// Adding a : Ring bound makes it not ICE
fn gcd<T>() {
    map_coeff::<T>(|_: u16| {});
}

// Removing the : Ring bound makes it not ICE
pub fn bivariate_factorization<T: Ring>() {
    gcd::<T>();
}
