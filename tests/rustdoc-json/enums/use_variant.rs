//@ arg always_none .index[] | select(.name == "AlwaysNone")
pub enum AlwaysNone {
    //@ arg none .index[] | select(.name == "None").id
    None,
}
//@ jq $always_none.inner.enum.variants? == [$none]

//@ arg use_none .index[] | select(.inner.use)
//@ jq $use_none.inner.use.id? == $none
pub use AlwaysNone::None;

//@ jq .index["\(.root)"].inner.module.items? == [$always_none.id, $use_none.id]
