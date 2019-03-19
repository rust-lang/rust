// ignore-emscripten
// ignore-s390x
// ignore-powerpc
// ignore-powerpc64
// ignore-powerpc64le
// ignore-sparc
// ignore-sparc64
// ignore-mips
// ignore-mips64
// ignore-aarch64
// ignore-arm

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

pub unsafe fn bar() { foo() }

extern "C" {
    // CHECK-LABEL: declare void @foo()
    // CHECK-SAME: [[ATTRS:#[0-9]+]]
    // CHECK-DAG: attributes [[ATTRS]] = { {{.*}}"target-features"="+avx2"{{.*}} }
    #[target_feature(enable = "avx2")]
    pub fn foo();
}
