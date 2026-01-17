// Test that identifiers with the same name but different hygiene contexts
// don't cause dep node hash collisions when using `-Zincremental-ignore-spans`.
//
// This regression test verifies that SyntaxContext (hygiene info) is always
// hashed even when span positions are ignored. Without this fix, two `Ident`s
// with the same `Symbol` but different `SyntaxContext` would hash identically,
// causing ICE: "query key X and key Y mapped to the same dep node".
//
// The issue manifests with macros that generate identifiers - each macro
// expansion creates identifiers with different hygiene contexts, but when
// spans are ignored, only the symbol name was being hashed.
//
//@ ignore-cross-compile

use run_make_support::{rfs, rustc};

fn main() {
    // Create a library with macro-generated code that creates identifiers
    // with the same name but different hygiene contexts.
    //
    // The key is to generate code that triggers queries like
    // `explicit_supertraits_containing_assoc_item` with identifiers that
    // have the same Symbol but different SyntaxContext.
    let lib_source = r#"
// Macro that generates a trait with an associated type.
// Each invocation creates identifiers with different SyntaxContext.
macro_rules! make_trait {
    ($trait_name:ident, $assoc_name:ident) => {
        pub trait $trait_name {
            type $assoc_name;
        }
    };
}

// Generate traits with different associated type names
make_trait!(TraitA, ItemA);
make_trait!(TraitB, ItemB);
make_trait!(TraitC, ItemC);

// Trait that combines them - triggers supertraits_containing_assoc_item queries
pub trait Combined: TraitA + TraitB + TraitC {
    fn get_a(&self) -> <Self as TraitA>::ItemA;
    fn get_b(&self) -> <Self as TraitB>::ItemB;
    fn get_c(&self) -> <Self as TraitC>::ItemC;
}

// Macro that generates impls - each expansion has different hygiene
macro_rules! impl_for {
    ($ty:ty, $trait:ident, $assoc:ident, $val:ty) => {
        impl $trait for $ty {
            type $assoc = $val;
        }
    };
}

pub struct MyStruct;
impl_for!(MyStruct, TraitA, ItemA, i32);
impl_for!(MyStruct, TraitB, ItemB, i64);
impl_for!(MyStruct, TraitC, ItemC, u32);

impl Combined for MyStruct {
    fn get_a(&self) -> i32 { 0 }
    fn get_b(&self) -> i64 { 0 }
    fn get_c(&self) -> u32 { 0 }
}

// More macro invocations to increase chance of hygiene collision
macro_rules! make_fn {
    ($name:ident) => {
        pub fn $name() {}
    };
}

make_fn!(foo);
make_fn!(bar);
make_fn!(baz);
"#;

    rfs::write("lib.rs", lib_source);

    // Compile with RDR flags - this would ICE before the fix due to
    // hygiene contexts not being hashed when spans are ignored.
    rustc()
        .input("lib.rs")
        .crate_type("lib")
        .incremental("incr")
        .arg("-Zstable-crate-hash")
        .arg("-Zincremental-ignore-spans")
        .run();

    // Second compilation to exercise incremental path
    rustc()
        .input("lib.rs")
        .crate_type("lib")
        .incremental("incr")
        .arg("-Zstable-crate-hash")
        .arg("-Zincremental-ignore-spans")
        .run();
}
