// only-x86_64
// compile-flags: -C target-feature=+avx

#![crate_type = "lib"]

#[no_mangle]
pub fn foo() {
    // CHECK: attributes #0 = { {{.*}}"target-features"="+avx"{{.*}} }
}
