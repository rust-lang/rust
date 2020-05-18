// only-x86_64
// compile-flags: -C target-feature=+sse2,-avx,+avx2 -C target-feature=+avx,-avx2

#![crate_type = "lib"]

#[no_mangle]
pub fn foo() {
    // CHECK: attributes #0 = { {{.*}}"target-features"="+sse2,-avx,+avx2,+avx,-avx2"{{.*}} }
}
