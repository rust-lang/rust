// FIXME(fmease): Re-audit this test! Rewrite its description! Probably rename the entire file, too!

// Demonstrate that enabling `generic_const_args` changes the behavior for trait object types in
// (unchecked) type aliases where the corresponding trait has non-type associated consts:
// The associated const must be specified.

//@ revisions: no_gca gca
//@ compile-flags: -Znext-solver=globally

#![cfg_attr(gca, feature(generic_const_args, min_generic_const_args))]

//[no_gca]~v ERROR not dyn compatible
type TyAlias = dyn HasNonTypeAssocConst;
//[gca]~^ ERROR the value of the associated constant `N` in `HasNonTypeAssocConst` must be specified

trait HasNonTypeAssocConst {
    /*non-type */const N: usize;
}

fn main() {}
