//! This module contains structures for filtering the expected types.
//! Use case for structures in this module is, for example, situation when you need to process
//! only certain `Enum`s.

use std::iter;

use hir::Semantics;
use syntax::ast::{self, Pat, make};

use crate::RootDatabase;

/// Enum types that implement `std::ops::Try` trait.
#[derive(Clone, Copy, Debug)]
pub enum TryEnum {
    Result,
    Option,
}

impl TryEnum {
    const ALL: [TryEnum; 2] = [TryEnum::Option, TryEnum::Result];

    /// Returns `Some(..)` if the provided type is an enum that implements `std::ops::Try`.
    pub fn from_ty(sema: &Semantics<'_, RootDatabase>, ty: &hir::Type<'_>) -> Option<TryEnum> {
        let enum_ = match ty.as_adt() {
            Some(hir::Adt::Enum(it)) => it,
            _ => return None,
        };
        TryEnum::ALL.iter().find_map(|&var| {
            if enum_.name(sema.db).as_str() == var.type_name() {
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
                make::ext::ident_path("Err"),
                iter::once(make::wildcard_pat().into()),
            )
            .into(),
            TryEnum::Option => make::ext::simple_ident_pat(make::name("None")).into(),
        }
    }

    pub fn happy_pattern(self, pat: Pat) -> ast::Pat {
        match self {
            TryEnum::Result => {
                make::tuple_struct_pat(make::ext::ident_path("Ok"), iter::once(pat)).into()
            }
            TryEnum::Option => {
                make::tuple_struct_pat(make::ext::ident_path("Some"), iter::once(pat)).into()
            }
        }
    }

    pub fn happy_pattern_wildcard(self) -> ast::Pat {
        match self {
            TryEnum::Result => make::tuple_struct_pat(
                make::ext::ident_path("Ok"),
                iter::once(make::wildcard_pat().into()),
            )
            .into(),
            TryEnum::Option => make::tuple_struct_pat(
                make::ext::ident_path("Some"),
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
