//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// One of the minimizations from trait-system-refactor-initiative#257 which ended up
// getting fixed by #153614. It seems somewhat likely that this is somewhat
// accidental by changing the exact shape of the cycle, causing us to avoid the
// underlying issue of trait-system-refactor-initiative#257.

pub trait Field: Sized + HasUnderlier<Underlier: PackScalar<Self>> {}
pub trait PackScalar<F>: 'static + UnderlierType {
    type Packed;
}
trait PackedField {
    type Scalar;
}
pub trait UnderlierType {}
pub trait HasUnderlier {
    type Underlier;
}
impl<U: UnderlierType> HasUnderlier for U {
    type Underlier = U;
}
impl UnderlierType for u8 {}
struct MyField;
impl Field for MyField {}
impl HasUnderlier for MyField {
    type Underlier = u8;
}
impl<F> PackScalar<F> for u8
where
    F: Field,
{
    type Packed = PackedPrimitiveType<F>;
}
pub struct PackedPrimitiveType<Scalar: Field>(Scalar);
impl<Scalar> PackedField for PackedPrimitiveType<Scalar>
where
    Scalar: Field,
{
    type Scalar = Scalar;
}

pub trait PackedTransformationFactory<OP> {}
trait TaggedPackedTransformationFactory<OP>: PackedField<Scalar: Field> {}
impl<OP> PackedTransformationFactory<OP> for PackedPrimitiveType<MyField> where
    Self: TaggedPackedTransformationFactory<OP>
{
}
fn main() {}
