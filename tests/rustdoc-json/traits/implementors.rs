#![feature(no_core)]
#![no_core]

// @set wham = "$.index[*][?(@.name=='Wham')].id"
// @count "$.index[*][?(@.name=='Wham')].inner.implementations[*]" 1
// @set gmWham = "$.index[*][?(@.name=='Wham')].inner.implementations[0]"
pub trait Wham {}

// @count "$.index[*][?(@.name=='GeorgeMichael')].inner.impls[*]" 1
// @is "$.index[*][?(@.name=='GeorgeMichael')].inner.impls[0]" $gmWham
// @set gm = "$.index[*][?(@.name=='Wham')].id"

// jsonpath_lib isnt expressive enough (for now) to get the "impl" item, so we
// just check it isn't pointing to the type, but when you port to jsondocck-ng
// check what the impl item is
// @!is "$.index[*][?(@.name=='Wham')].inner.implementations[0]" $gm
pub struct GeorgeMichael {}

impl Wham for GeorgeMichael {}
