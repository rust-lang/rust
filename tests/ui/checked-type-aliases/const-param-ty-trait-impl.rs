// FIXME use expand_free_alias_tys in coherence::builtin ??

#![feature(checked_type_aliases, const_param_ty_trait)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy_;


#[derive(PartialEq, Eq)]
pub struct ConstParamTyLocal {
    pub foo: u32,
}

pub type ConstParamTyLocalAlias = ConstParamTyLocal;
impl ConstParamTy_ for ConstParamTyLocalAlias { }
//~^ERROR: the trait `ConstParamTy` may not be implemented for this type
fn main() {}
