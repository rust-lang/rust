// compile-flags: -C no-prepopulate-passes -Zshare-generics=yes

// Check that local generics are internalized if they are in the same CGU

// CHECK: define internal {{.*}} @_ZN34local_generics_in_exe_internalized3foo{{.*}}
pub fn foo<T>(x: T, y: T) -> (T, T) {
    (x, y)
}

fn main() {
    let _ = foo(0u8, 1u8);
}
