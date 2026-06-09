//@build-pass
// We currently accept conflicting associated type bounds with different generics,
// which results in some weirdness.
// See https://github.com/rust-lang/rust/issues/154662

trait Dummy {
    type DummyAssoc1;
    type DummyAssoc2;
}
struct DummyStruct;
impl Dummy for DummyStruct {
    type DummyAssoc1 = i16;
    type DummyAssoc2 = i16;
}

trait Super<T> {
    type Assoc;
}

trait Sub<D: Dummy>: Super<D::DummyAssoc1, Assoc = i32> + Super<D::DummyAssoc2, Assoc = i64> {}

fn require_trait<D: Dummy, U: Super<D::DummyAssoc1> + ?Sized>() {}

fn use_dyn<D: Dummy>() {
    require_trait::<D, dyn Sub<D>>();
}

fn main() {
    // This ends up proving that `dyn Sub<DummyStruct>` implements `Super<i16>`.
    // However, `dyn Sub<DummyStruct>` has bounds for both `Assoc = i32` and `Assoc = i64`,
    // which is nonsense.
    use_dyn::<DummyStruct>();
}
