//@ compile-flags: -Z annotate-moves=8 -C debuginfo=full
//@ ignore-std-debug-assertions
//@ edition: 2021

#![crate_type = "lib"]

// Test impl Trait with a large iterator type
fn make_large_iter() -> impl Iterator<Item = u64> {
    fn double(x: u64) -> u64 {
        x * 2
    }

    // Chain<Map<IntoIter<u64, 5>, fn(u64) -> u64>, Map<IntoIter<u64, 5>, fn(u64) -> u64>>
    // IntoIter owns the array data, making the iterator larger
    [1u64, 2, 3, 4, 5]
        .into_iter()
        .map(double as fn(u64) -> u64)
        .chain([6u64, 7, 8, 9, 10].into_iter().map(double as fn(u64) -> u64))
}

// EMIT_MIR iter.test_impl_trait_return.AnnotateMoves.after.mir
pub fn test_impl_trait_return() -> impl Iterator<Item = u64> {
    // CHECK-LABEL: fn test_impl_trait_return(
    // The iterator is returned directly without moving through a local
    make_large_iter()
}

// EMIT_MIR iter.test_impl_trait_arg.AnnotateMoves.after.mir
pub fn test_impl_trait_arg(iter: impl Iterator<Item = u64>) -> Vec<u64> {
    // CHECK-LABEL: fn test_impl_trait_arg(
    // Generic impl trait parameter - concrete type determined at call site
    iter.collect()
}

// EMIT_MIR iter.test_impl_trait_chain.AnnotateMoves.after.mir
pub fn test_impl_trait_chain() -> Vec<u64> {
    // CHECK-LABEL: fn test_impl_trait_chain(
    // The iterator move shows up when passing to test_impl_trait_arg
    // CHECK: scope {{[0-9]+}} (inlined core::profiling::compiler_move::<std::iter::Chain<Map<std::array::IntoIter<u64, 5>, fn(u64) -> u64>, Map<std::array::IntoIter<u64, 5>, fn(u64) -> u64>>, {{[0-9]+}}>)
    let iter = make_large_iter();
    test_impl_trait_arg(iter)
}
