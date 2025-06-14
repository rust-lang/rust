#![feature(no_core, lang_items)]
#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
#![no_core]
#![rustc_coherence_is_core]

//@ set impl_i32 = "$.index[?(@.docs=='Only core can do this')].id"

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

/// Only core can do this
impl i32 {
    //@ set identity = "$.index[?(@.docs=='Do Nothing')].id"

    /// Do Nothing
    pub fn identity(self) -> Self {
        self
    }

    //@ is "$.index[?(@.docs=='Only core can do this')].inner.impl.items[*]" $identity
}

//@ set Trait = "$.index[?(@.name=='Trait')].id"
pub trait Trait {}
//@ set impl_trait_for_i32 = "$.index[?(@.docs=='impl Trait for i32')].id"
/// impl Trait for i32
impl Trait for i32 {}

/// i32
#[rustc_doc_primitive = "i32"]
mod prim_i32 {}

//@ set i32 = "$.index[?(@.docs=='i32')].id"
//@ is "$.index[?(@.docs=='i32')].name" '"i32"'
//@ is "$.index[?(@.docs=='i32')].inner.primitive.name" '"i32"'
//@ ismany "$.index[?(@.docs=='i32')].inner.primitive.impls[*]" $impl_i32 $impl_trait_for_i32
