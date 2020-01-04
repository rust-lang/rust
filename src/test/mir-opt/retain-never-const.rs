// Regression test for #66975 - ensure that we don't keep unevaluated
// `!`-typed constants until codegen.

// Force generation of optimized mir for functions that do not reach codegen.
// compile-flags: --emit mir,link

#![feature(const_panic)]
#![feature(never_type)]
#![warn(const_err)]

struct PrintName<T>(T);

impl<T> PrintName<T> {
    const VOID: ! = panic!();
}

fn no_codegen<T>() {
    let _ = PrintName::<T>::VOID;
}

fn main() {}

// END RUST SOURCE
// START rustc.no_codegen.PreCodegen.after.mir
// bb0: {
//     StorageLive(_1);
//     _1 = const PrintName::<T>::VOID;
//     unreachable;
// }
// END rustc.no_codegen.PreCodegen.after.mir
