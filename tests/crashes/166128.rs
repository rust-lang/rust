//@ known-bug: #166128
//@ compile-flags: -Znext-solver=globally

// const_param_ty_unchecked is internal-only and skips the ConstParamTy_ check that would
// otherwise reject fn() here (#161131), so the function pointer reaches valtree conversion,
// which fails without any error having been reported.

#![feature(gca_min_const_items)]
#![feature(adt_const_params)]
#![feature(gca_const_items)]
#![feature(const_param_ty_unchecked)]

trait Trait {
    const F: fn();
}

fn take<T>()
where
    T: Trait<F = { || {} }>,
{
}

fn main() {}
