//@ revisions: O2 S Z
//@[O2] compile-flags: -Copt-level=2
//@[S] compile-flags: -Copt-level=s
//@[Z] compile-flags: -Copt-level=z
//@ ignore-wasi wasi codegens the main symbol differently

// The C entry shim has no item behind it, so it does not go through the per-item attribute path.
// It still has to carry the size attributes, or `main` is the one definition in a size build that
// LLVM optimizes for speed.

#![crate_type = "bin"]

// CHECK-LABEL: define {{.*}} @main(
// CHECK-SAME: [[MAIN_ATTRS:#[0-9]+]]

// O2: attributes [[MAIN_ATTRS]] =
// O2-NOT: optsize
// O2-NOT: minsize

// S: attributes [[MAIN_ATTRS]] =
// S-NOT: minsize
// S-SAME: optsize

// Z: attributes [[MAIN_ATTRS]] =
// Z-SAME: minsize
// Z-SAME: optsize

fn main() {
    std::hint::black_box(0);
}
