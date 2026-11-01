// gca! is a built-in macro relevant to gca_min_const_items; its contents should be formatted as
// if its contents were passed through unchanged (the macro changes the semantics of the contained
// expression, the syntax is unchanged)

#![feature(gca_min_const_items)]

use std::gca;

trait Trait {
    #[rustc_always_gca]
    const TYPE_CONST: usize;
}

struct S<const N: usize>;

fn parsed_as_expr_kind<T: Trait>(_: S<{ gca!(T::TYPE_CONST) }>) {}
fn parsed_as_ty_kind<T: Trait>(_: S<gca!(T::TYPE_CONST)>) {}
