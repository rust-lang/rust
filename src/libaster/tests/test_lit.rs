// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;
use syntax::codemap::{DUMMY_SP, Spanned};
use syntax::ptr::P;

use super::super::AstBuilder;

#[test]
fn test_int() {
    let builder = AstBuilder::new();

    fn check(lit: P<ast::Lit>, value: u64, lit_int_ty: ast::LitIntType) {
        assert_eq!(
            lit,
            P(Spanned {
                span: DUMMY_SP,
                node: ast::LitInt(value, lit_int_ty),
            })
        );
    }

    check(builder.lit().i8(1), 1, ast::SignedIntLit(ast::TyI8, ast::Plus));
    check(builder.lit().i16(1), 1, ast::SignedIntLit(ast::TyI16, ast::Plus));
    check(builder.lit().i32(1), 1, ast::SignedIntLit(ast::TyI32, ast::Plus));
    check(builder.lit().i64(1), 1, ast::SignedIntLit(ast::TyI64, ast::Plus));
    check(builder.lit().isize(1), 1, ast::SignedIntLit(ast::TyIs, ast::Plus));

    check(builder.lit().i8(-1), !0, ast::SignedIntLit(ast::TyI8, ast::Minus));
    check(builder.lit().i16(-1), !0, ast::SignedIntLit(ast::TyI16, ast::Minus));
    check(builder.lit().i32(-1), !0, ast::SignedIntLit(ast::TyI32, ast::Minus));
    check(builder.lit().i64(-1), !0, ast::SignedIntLit(ast::TyI64, ast::Minus));
    check(builder.lit().isize(-1), !0, ast::SignedIntLit(ast::TyIs, ast::Minus));

    check(builder.lit().u8(1), 1, ast::UnsignedIntLit(ast::TyU8));
    check(builder.lit().u16(1), 1, ast::UnsignedIntLit(ast::TyU16));
    check(builder.lit().u32(1), 1, ast::UnsignedIntLit(ast::TyU32));
    check(builder.lit().u64(1), 1, ast::UnsignedIntLit(ast::TyU64));
    check(builder.lit().usize(1), 1, ast::UnsignedIntLit(ast::TyUs));

    check(builder.lit().int(1), 1, ast::UnsuffixedIntLit(ast::Plus));
    check(builder.lit().int(-1), !0, ast::UnsuffixedIntLit(ast::Minus));
}

#[test]
fn test_bool() {
    let builder = AstBuilder::new();

    assert_eq!(builder.lit().bool(true),
        P(Spanned {
            span: DUMMY_SP,
            node: ast::LitBool(true)
        })
    );

    assert_eq!(builder.lit().bool(false),
        P(Spanned {
            span: DUMMY_SP,
            node: ast::LitBool(false)
        })
    );
}

#[test]
fn test_str() {
    let builder = AstBuilder::new();

    assert_eq!(builder.lit().str("string"),
        P(Spanned {
            span: DUMMY_SP,
            node: ast::LitStr(
                builder.interned_string("string"),
                ast::CookedStr,
            ),
        })
    );
}
