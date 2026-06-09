#![feature(impl_trait_in_assoc_type)]

pub struct AlwaysTrue;

/// impl IntoIterator
impl IntoIterator for AlwaysTrue {
    //@ set Item = '$.index[?(@.docs=="type Item")].id'
    /// type Item
    type Item = bool;

    //@ count '$.index[?(@.docs=="type IntoIter")].inner.assoc_type.type.impl_trait[*]' 1
    //@ is    '$.index[?(@.docs=="type IntoIter")].inner.assoc_type.type.impl_trait[0].trait_bound.trait.path' '"Iterator"'
    //@ count '$.index[?(@.docs=="type IntoIter")].inner.assoc_type.type.impl_trait[0].trait_bound.trait.args.angle_bracketed.constraints[*]' 1
    //@ is    '$.index[?(@.docs=="type IntoIter")].inner.assoc_type.type.impl_trait[0].trait_bound.trait.args.angle_bracketed.constraints[0].name' '"Item"'
    //@ is    '$.index[?(@.docs=="type IntoIter")].inner.assoc_type.type.impl_trait[0].trait_bound.trait.args.angle_bracketed.constraints[0].binding.equality.type.primitive' '"bool"'

    //@ set IntoIter = '$.index[?(@.docs=="type IntoIter")].id'
    /// type IntoIter
    type IntoIter = impl Iterator<Item = bool>;

    //@ set into_iter = '$.index[?(@.docs=="fn into_iter")].id'
    /// fn into_iter
    fn into_iter(self) -> Self::IntoIter {
        std::iter::repeat(true)
    }
}

//@ ismany '$.index[?(@.docs=="impl IntoIterator")].inner.impl.items[*]' $Item $IntoIter $into_iter
