// Verifies that type metadata identifiers for drop functions are emitted correctly.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Copt-level=0 -Zsanitizer=cfi -Ctarget-feature=-crt-static

#![crate_type = "lib"]

// CHECK-LABEL: define{{.*}}4core3ptr47drop_in_place$LT$dyn$u20$core..marker..Send$GT$
// CHECK-SAME:  {{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
// CHECK:       call i1 @llvm.type.test(ptr {{%.+}}, metadata !"_ZTSFvPu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops4drop4Dropu6regionEE")

struct EmptyDrop;
// CHECK: define{{.*}}4core3ptr{{[0-9]+}}drop_in_place$LT${{.*}}EmptyDrop$GT${{.*}}!type ![[TYPE1]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

struct NonEmptyDrop;

impl Drop for NonEmptyDrop {
    fn drop(&mut self) {}
    // CHECK: define{{.*}}4core3ptr{{[0-9]+}}drop_in_place$LT${{.*}}NonEmptyDrop$GT${{.*}}!type ![[TYPE1]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
}

pub fn foo() {
    let _ = Box::new(EmptyDrop) as Box<dyn Send>;
    let _ = Box::new(NonEmptyDrop) as Box<dyn Send>;
}

// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvPu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops4drop4Dropu6regionEE"}
