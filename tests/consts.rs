#![allow(plugin_as_library)]
#![feature(rustc_private)]

extern crate clippy;
extern crate syntax;
extern crate rustc;

use clippy::consts::constant;
use clippy::consts::ConstantVariant::*;
use syntax::ast::*;
use syntax::ptr::P;
use syntax::codemap::{Spanned, COMMAND_LINE_SP};
use std::mem;
use rustc::lint::Context;

fn ctx() -> &'static Context<'static, 'static> {
	unsafe { 
		let x : *const Context<'static, 'static> = std::ptr::null();
		mem::transmute(x)
	}
}

#[test]
fn test_lit() {
    assert_eq!(Some(ConstantBool(true)), constant(ctx(),
        &Expr{ 
			id: 1, 
			node: ExprLit(P(Spanned{ 
				node: LitBool(true), 
				span: COMMAND_LINE_SP,
			})), 
			span: COMMAND_LINE_SP,
		}).map(|x| x.constant));
}
