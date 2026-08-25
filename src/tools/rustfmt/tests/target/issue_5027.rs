// rustfmt-style_edition: 2024

pub type Iter<'a, D> = impl DoubleEndedIterator<Item = (SomethingSomethingSomethingLongType<D>)>
    + ExactSizeIterator
    + 'a;

trait FOo {
    pub type Iter<'a, D> = impl DoubleEndedIterator<Item = (SomethingSomethingSomethingLongType<D>)>
        + ExactSizeIterator
        + 'a;
}

impl Bar {
    type Iter<'a, D> = impl DoubleEndedIterator<Item = (SomethingSomethingSomethingLongType<D>)>
        + ExactSizeIterator
        + 'a;
}
