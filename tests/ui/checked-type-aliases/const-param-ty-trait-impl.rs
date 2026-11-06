//@ check-pass

#![feature(checked_type_aliases, const_param_ty_trait)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy_;

#[derive(PartialEq, Eq)]
pub struct ConstParamTyLocal {
    pub foo: u32,
}

pub type ConstParamTyLocalAlias = ConstParamTyLocal;
impl ConstParamTy_ for ConstParamTyLocalAlias { }

fn main() {}
