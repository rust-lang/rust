// Verifies that type metadata identifiers for drop functions are emitted correctly.
//
// Non needs_drop drop glue isn't codegen'd at all, so we don't try to check the IDs there. But we
// do check it's not emitted which should help catch bugs if we do start generating it again in the
// future.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Copt-level=0 -Zsanitizer=cfi -Ctarget-feature=-crt-static

#![crate_type = "lib"]

// CHECK-LABEL: define{{.*}}4core3ptr47drop_in_place$LT$dyn$u20$core..marker..Send$GT$
// CHECK-SAME:  {{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
// CHECK:       call i1 @llvm.type.test(ptr {{%.+}}, metadata !"_ZTSFvPu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops4drop4Dropu6regionEE")

struct EmptyDrop;
// CHECK-NOT: define{{.*}}4core3ptr{{[0-9]+}}drop_in_place$LT${{.*}}EmptyDrop$GT${{.*}}!type ![[TYPE1]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

struct PresentDrop;

impl Drop for PresentDrop {
    fn drop(&mut self) {}
    // CHECK: define{{.*}}4core3ptr{{[0-9]+}}drop_in_place$LT${{.*}}PresentDrop$GT${{.*}}!type ![[TYPE1]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
}

pub fn foo() {
    let _ = Box::new(EmptyDrop) as Box<dyn Send>;
    let _ = Box::new(PresentDrop) as Box<dyn Send>;
}

// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvPu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops4drop4Dropu6regionEE"}
