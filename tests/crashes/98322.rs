//@ known-bug: #98322

#![feature(generic_const_exprs)]

// Main function seems irrelevant
fn main() {}

// Constant must be provided via an associated constant in a trait
pub trait ConstTrait {
    const ASSOC_CONST: usize;
}

// For some reason I find it's necessary to have an implementation of this trait that recurses
pub trait OtherTrait
{
    fn comm(self);
}

// There must be a blanket impl here
impl<T> OtherTrait for T where
    T: ConstTrait,
    [();T::ASSOC_CONST]: Sized,
{
    fn comm(self) {
        todo!()
    }
}

// The struct must be recursive
pub struct RecursiveStruct(Box<RecursiveStruct>);

// This implementation must exist, and it must recurse into its child
impl OtherTrait for RecursiveStruct {
    fn comm(self) {
        (self.0).comm();
    }
}
