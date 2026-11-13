//@ check-pass
// Make sure that the dyn-compatibility check for conflicting bounds
// only use bounds from supertraits, and not from type parameters.

pub trait HasAssoc<T: ?Sized> {
    type Assoc;
}

pub trait Sub: HasAssoc<Self, Assoc = Self> + HasAssoc<i32, Assoc = i64> {}

pub trait Trait<D: Sub> {}

pub fn what<D: Sub>(_: &dyn Trait<D>) {}

fn main() {}
