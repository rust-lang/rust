/// Remaps the type with a different lifetime for 'tcx if applicable.
pub trait Remap {
    type Remap<'a>;
}

impl Remap for u32 {
    type Remap<'a> = u32;
}

impl<T: Remap> Remap for Option<T> {
    type Remap<'a> = Option<T::Remap<'a>>;
}

impl Remap for () {
    type Remap<'a> = ();
}

impl<T0: Remap, T1: Remap> Remap for (T0, T1) {
    type Remap<'a> = (T0::Remap<'a>, T1::Remap<'a>);
}

impl<T0: Remap, T1: Remap, T2: Remap> Remap for (T0, T1, T2) {
    type Remap<'a> = (T0::Remap<'a>, T1::Remap<'a>, T2::Remap<'a>);
}

impl<T0: Remap, T1: Remap, T2: Remap, T3: Remap> Remap for (T0, T1, T2, T3) {
    type Remap<'a> = (T0::Remap<'a>, T1::Remap<'a>, T2::Remap<'a>, T3::Remap<'a>);
}
