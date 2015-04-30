//! Checks for incompatible bit masks in comparisons, e.g. `x & 1 == 2`. This cannot work because the bit that makes up
//! the value two was zeroed out by the bit-and with 1. So the formula for detecting if an expression of the type 
//! `_ <bit_op> m <cmp_op> c` (where `<bit_op>` is one of {`&`, '|'} and `<cmp_op>` is one of {`!=`, `>=`, `>` ,`!=`, `>=`, 
//! `>`}) can be determined from the following table:
//! 
//! |Comparison  |Bit-Op|Example     |is always|Formula               |
//! |------------|------|------------|---------|----------------------|
//! |`==` or `!=`| `&`  |`x & 2 == 3`|`false`  |`c & m != c`          |
//! |`<`  or `>=`| `&`  |`x & 2 < 3` |`true`   |`m < c`               |
//! |`>`  or `<=`| `&`  |`x & 1 > 1` |`false`  |`m <= c`              |
//! |`==` or `!=`| `|`  |`x | 1 == 0`|`false`  |`c | m != c`          |
//! |`<`  or `>=`| `|`  |`x | 1 < 1` |`false`  |`m >= c`			  |
//! |`<=` or `>` | `|`  |`x | 1 > 0` |`true`   |`m > c`               |
//! 
//! *TODO*: There is the open question if things like `x | 1 > 1` should be caught by this lint, because it is basically
//! an obfuscated version of `x > 1`.
//! 
//! This lint is **deny** by default

use rustc::plugin::Registry;
use rustc::lint::*;
use rustc::middle::const_eval::lookup_const_by_id;
use rustc::middle::def::*;
use syntax::ast::*;
use syntax::ast_util::{is_comparison_binop, binop_to_string};
use syntax::ptr::P;
use syntax::codemap::Span;

declare_lint! {
    pub BAD_BIT_MASK,
    Deny,
    "Deny the use of incompatible bit masks in comparisons, e.g. '(a & 1) == 2'"
}

#[derive(Copy,Clone)]
pub struct BitMask;

impl LintPass for BitMask {
    fn get_lints(&self) -> LintArray {
        lint_array!(BAD_BIT_MASK)
    }
    
    fn check_expr(&mut self, cx: &Context, e: &Expr) {
        if let ExprBinary(ref cmp, ref left, ref right) = e.node {
			if is_comparison_binop(cmp.node) {
				fetch_int_literal(cx, right).map(|cmp_value| check_compare(cx, left, cmp.node, cmp_value, &e.span));
			}
		}
    }
}

fn check_compare(cx: &Context, bit_op: &Expr, cmp_op: BinOp_, cmp_value: u64, span: &Span) {
	match &bit_op.node {
		&ExprParen(ref subexp) => check_compare(cx, subexp, cmp_op, cmp_value, span),
		&ExprBinary(ref op, ref left, ref right) => {
			if op.node != BiBitAnd && op.node != BiBitOr { return; }
			if let Some(mask_value) = fetch_int_literal(cx, right) {
				check_bit_mask(cx, op.node, cmp_op, mask_value, cmp_value, span);
			} else if let Some(mask_value) = fetch_int_literal(cx, left) {
				check_bit_mask(cx, op.node, cmp_op, mask_value, cmp_value, span);
			}
		},
		_ => ()
	}
}

fn check_bit_mask(cx: &Context, bit_op: BinOp_, cmp_op: BinOp_, mask_value: u64, cmp_value: u64, span: &Span) {
	match cmp_op {
		BiEq | BiNe => match bit_op {
			BiBitAnd => if mask_value & cmp_value != mask_value {
				cx.span_lint(BAD_BIT_MASK, *span, &format!("incompatible bit mask: _ & {} can never be equal to {}", mask_value,
					cmp_value));
			},
			BiBitOr => if mask_value | cmp_value != cmp_value {
				cx.span_lint(BAD_BIT_MASK, *span, &format!("incompatible bit mask: _ | {} can never be equal to {}", mask_value,
					cmp_value));
			},
			_ => ()
		},
		BiLt | BiGe => match bit_op {
			BiBitAnd => if mask_value < cmp_value {
				cx.span_lint(BAD_BIT_MASK, *span, &format!("incompatible bit mask: _ & {} will always be lower than {}", mask_value,
					cmp_value));
			},
			BiBitOr => if mask_value >= cmp_value {
				cx.span_lint(BAD_BIT_MASK, *span, &format!("incompatible bit mask: _ | {} will never be lower than {}", mask_value,
					cmp_value));
			},
			_ => ()
		},
		BiLe | BiGt => match bit_op {
			BiBitAnd => if mask_value <= cmp_value {
				cx.span_lint(BAD_BIT_MASK, *span, &format!("incompatible bit mask: _ & {} will never be higher than {}", mask_value,
					cmp_value));
			},
			BiBitOr => if mask_value > cmp_value {
				cx.span_lint(BAD_BIT_MASK, *span, &format!("incompatible bit mask: _ | {} will always be higher than {}", mask_value,
					cmp_value));				
			},
			_ => ()
		},
		_ => ()
	}
}

fn fetch_int_literal(cx: &Context, lit : &Expr) -> Option<u64> {
	match &lit.node {
		&ExprLit(ref lit_ptr) => {
			if let &LitInt(value, _) = &lit_ptr.node {
				Option::Some(value) //TODO: Handle sign
			} else { Option::None }
		},
		&ExprPath(_, _) => {
			let def_map = cx.tcx.def_map.borrow();
			let path_res_op = def_map.get(&lit.id);
			path_res_op.as_ref().and_then(|x| {
				if let &DefConst(def_id) = &x.base_def {
					lookup_const_by_id(cx.tcx, def_id, Option::None).and_then(|l| fetch_int_literal(cx, l))
				} else { Option::None }
			})
		},
		_ => Option::None
	}
}
