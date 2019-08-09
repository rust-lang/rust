// compile-flags: -C no-prepopulate-passes -Zshare-generics=yes

// Check that local generics are internalized if they are in the same CGU

// CHECK-LABEL: ; local_generics_in_exe_internalized::foo
// CHECK-NEXT: ; Function Attrs:
// CHECK-NEXT: define internal
pub fn foo<T>(x: T, y: T) -> (T, T) {
    (x, y)
}

fn main() {
    let _ = foo(0u8, 1u8);
}
