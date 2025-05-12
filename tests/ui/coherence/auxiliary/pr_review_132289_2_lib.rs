pub type A = &'static [usize; 1];
pub type B = &'static [usize; 100];

pub trait Trait<P> {
    type Assoc;
}

pub type Dyn<P> = dyn Trait<P, Assoc = A>;

pub trait LocallyUnimplemented<P> {}

impl<P, T: ?Sized> Trait<P> for T
where
    T: LocallyUnimplemented<P>,
{
    type Assoc = B;
}

trait MakeArray<Arr> {
    fn make() -> &'static Arr;
}
impl<const N: usize> MakeArray<[usize; N]> for () {
    fn make() -> &'static [usize; N] {
        &[1337; N]
    }
}

// it would be sound for this return type to be interpreted as being
// either of A or B, if that's what a soundness fix for overlap of
// dyn Trait's impls would entail

// In this test, we check at the call-site that the interpretation
// is consistent across crates in this specific scenario.
pub fn function<P>() -> (<Dyn<P> as Trait<P>>::Assoc, usize) {
    let val = <() as MakeArray<_>>::make();
    (val, val.len())
}
