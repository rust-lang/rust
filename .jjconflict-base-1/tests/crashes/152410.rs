//@ known-bug: #152410
trait Trait {
    fn foo(&self);
}
impl Trait for () {}

const OBJECT: *const (dyn Trait + Send) = &();

const _: *const dyn Send = OBJECT as _;
