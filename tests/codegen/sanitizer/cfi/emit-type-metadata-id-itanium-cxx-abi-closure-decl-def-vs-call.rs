// Verifies that type metadata identifiers for closure declaration/definition FnAbis are
// disambiguated from closure call FnAbis.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Copt-level=0 -Zsanitizer=cfi -Ctarget-feature=-crt-static

#![crate_type="lib"]

pub fn foo(f: fn()) {
    // CHECK-LABEL: define{{.*}}3foo
    // CHECK-SAME:  {{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
    // CHECK:       call i1 @llvm.type.test(ptr {{%.+}}, metadata !"_ZTSFvvE")
    f();
}


// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvPFvvEE"}
