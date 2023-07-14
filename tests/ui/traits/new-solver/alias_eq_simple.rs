// check-pass
// compile-flags: -Ztrait-solver=next
// compile-flags: -Zdump-solver-proof-tree=always
// compile-flags: -Zdump-solver-proof-tree-use-cache=no

#![feature(rustc_attrs)]
#![rustc_filter_proof_tree_dump(
    "Binder { value: TraitPredicate(<(i32, u32) as TraitA>, polarity:Positive), bound_vars: [] }"
)]

// test that the new solver can handle `alias-eq(<i32 as TraitB>::Assoc, u32)`

trait TraitA {}

trait TraitB {
    type Assoc;
}

impl<T: TraitB> TraitA for (T, T::Assoc) {}

impl TraitB for i32 {
    type Assoc = u32;
}

fn needs_a<T: TraitA>() {}

fn main() {
    needs_a::<(i32, u32)>();
}
