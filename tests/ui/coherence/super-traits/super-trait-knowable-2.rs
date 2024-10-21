// A regression test for pyella-0.1.5 which broke when
// enabling the new solver in coherence.
//
// `Tensor: TensorValue` is knowable while `Tensor: TensorOp<?t2>`
// may be implemented downstream. We previously didn't check the
// super trait bound in coherence, causing these impls to overlap.
//
// However, we did fail to normalize `<Tensor as TensorValue::Unmasked`
// which caused the old solver to emit a `Tensor: TensorValue` goal in
// `fn normalize_to_error` which then failed, causing this test to pass.

//@ check-pass

pub trait TensorValue {
    type Unmasked;
}

trait TensorCompare<T> {}
pub trait TensorOp<T>: TensorValue {}

pub struct Tensor;
impl<T2> TensorCompare<T2> for Tensor {}
impl<T1, T2> TensorCompare<T2> for T1
where
    T1: TensorOp<T2>,
    T1::Unmasked: Sized,
{}


fn main() {}
