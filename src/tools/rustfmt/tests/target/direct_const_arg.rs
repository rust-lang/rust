// direct_const_arg! is a built-in macro relevant to min_generic_const_args; its contents should be
// formatted as if its contents were passed through unchanged (the macro changes the semantics of
// the contained expression, the syntax is unchanged)

#![feature(min_generic_const_args)]

trait Trait {
    type const TYPE_CONST: usize;
}

struct S<const N: usize>;

fn parsed_as_expr_kind<T: Trait>(_: S<{ core::direct_const_arg!(T::TYPE_CONST) }>) {}
fn parsed_as_ty_kind<T: Trait>(_: S<core::direct_const_arg!(T::TYPE_CONST)>) {}
