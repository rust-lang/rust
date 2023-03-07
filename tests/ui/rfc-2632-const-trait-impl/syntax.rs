// compile-flags: -Z parse-only
// check-pass

#![feature(const_trait_bound_opt_out)]
#![feature(const_trait_impl)]

// For now, this parses since an error does not occur until AST lowering.
impl ~const T {}
