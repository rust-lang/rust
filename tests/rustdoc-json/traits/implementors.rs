pub trait Wham {}
pub struct GeorgeMichael {}

/// Wham for George Michael
impl Wham for GeorgeMichael {}

// Find IDs.
//@ set wham = "$.index[?(@.name=='Wham')].id"
//@ set gmWham = "$.index[?(@.docs=='Wham for George Michael')].id"
//@ set gm = "$.index[?(@.name=='GeorgeMichael')].id"

// Both struct and trait point to impl.
//@ has "$.index[?(@.name=='GeorgeMichael')].inner.struct.impls[*]" $gmWham
//@ is "$.index[?(@.name=='Wham')].inner.trait.implementations[*]" $gmWham

// Impl points to both struct and trait.
//@ is "$.index[?(@.docs == 'Wham for George Michael')].inner.impl.trait.id" $wham
//@ is "$.index[?(@.docs == 'Wham for George Michael')].inner.impl.for.resolved_path.id" $gm
