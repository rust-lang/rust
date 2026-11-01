//@ revisions: O2 S Z
//@[O2] compile-flags: -Copt-level=2
//@[S] compile-flags: -Copt-level=s
//@[Z] compile-flags: -Copt-level=z

// `s` gives `optsize` while `z` gives both `minsize` and `optsize`. This is the only coverage of
// the `-Copt-level` path; `optimize-attr-1.rs` reaches these via `#[optimize(size)]`.

#![crate_type = "lib"]

// CHECK-LABEL: @f(
// CHECK-SAME: [[F_ATTRS:#[0-9]+]]

// O2: attributes [[F_ATTRS]] =
// O2-NOT: optsize
// O2-NOT: minsize

// S: attributes [[F_ATTRS]] =
// S-NOT: minsize
// S-SAME: optsize

// Z: attributes [[F_ATTRS]] =
// Z-SAME: minsize
// Z-SAME: optsize

#[no_mangle]
pub fn f(x: u32) -> u32 {
    x + 1
}
