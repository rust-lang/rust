//! This module contains structures for filtering the expected types.
//! Use case for structures in this module is, for example, situation when you need to process
//! only certain `Enum`s.

use crate::RootDatabase;
use hir::{Adt, Semantics, Type};
use std::iter;
use syntax::ast::{self, make};

/// Enum types that implement `std::ops::Try` trait.
#[derive(Clone, Copy)]
pub enum TryEnum {
    Result,
    Option,
}

impl TryEnum {
    const ALL: [TryEnum; 2] = [TryEnum::Option, TryEnum::Result];

    /// Returns `Some(..)` if the provided type is an enum that implements `std::ops::Try`.
    pub fn from_ty(sema: &Semantics<RootDatabase>, ty: &Type) -> Option<TryEnum> {
        let enum_ = match ty.as_adt() {
            Some(Adt::Enum(it)) => it,
            _ => return None,
        };
        TryEnum::ALL.iter().find_map(|&var| {
            if &enum_.name(sema.db).to_string() == var.type_name() {
                return Some(var);
            }
            None
        })
    }

    pub fn happy_case(self) -> &'static str {
        match self {
            TryEnum::Result => "Ok",
            TryEnum::Option => "Some",
        }
    }

    pub fn sad_pattern(self) -> ast::Pat {
        match self {
            TryEnum::Result => make::tuple_struct_pat(
                make::path_unqualified(make::path_segment(make::name_ref("Err"))),
                iter::once(make::wildcard_pat().into()),
            )
            .into(),
            TryEnum::Option => make::ident_pat(make::name("None")).into(),
        }
    }

    pub fn happy_pattern(self) -> ast::Pat {
        match self {
            TryEnum::Result => make::tuple_struct_pat(
                make::path_unqualified(make::path_segment(make::name_ref("Ok"))),
                iter::once(make::wildcard_pat().into()),
            )
            .into(),
            TryEnum::Option => make::tuple_struct_pat(
                make::path_unqualified(make::path_segment(make::name_ref("Some"))),
                iter::once(make::wildcard_pat().into()),
            )
            .into(),
        }
    }

    fn type_name(self) -> &'static str {
        match self {
            TryEnum::Result => "Result",
            TryEnum::Option => "Option",
        }
    }
}
