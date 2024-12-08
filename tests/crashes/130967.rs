//@ known-bug: #130967

trait Producer {
    type Produced;
    fn make_one() -> Self::Produced;
}

impl<E: ?Sized> Producer for () {
    type Produced = Option<E>;
    fn make_one() -> Self::Produced {
        loop {}
    }
}
