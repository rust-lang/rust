#![feature(impl_trait_in_assoc_type)]

pub struct AlwaysTrue;

/// impl IntoIterator
impl IntoIterator for AlwaysTrue {
    //@ set Item = '$.index[?(@.docs=="type Item")].id'
    /// type Item
    type Item = bool;

    //@ is    '$.index[?(@.docs=="type IntoIter")].inner.assoc_type.type' 15
    //@ count '$.types[15].impl_trait[*]' 1
    //@ is    '$.types[15].impl_trait[0].trait_bound.trait.path' '"Iterator"'
    //@ count '$.types[15].impl_trait[0].trait_bound.trait.args.angle_bracketed.constraints[*]' 1
    //@ is    '$.types[15].impl_trait[0].trait_bound.trait.args.angle_bracketed.constraints[0].name' '"Item"'
    //@ is    '$.types[15].impl_trait[0].trait_bound.trait.args.angle_bracketed.constraints[0].binding.equality.type' 14
    //@ is    '$.types[14].primitive' '"bool"'

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
