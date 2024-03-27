// Verifies that type metadata identifiers for closures are emitted correctly.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi -Copt-level=0

#![crate_type="lib"]
pub fn foo1(a: fn(i32) -> i32) {
    // CHECK-LABEL: define{{.*}}4foo1{{.*}}!type !{{[0-9]+}}
    // CHECK:       call i1 @llvm.type.test(ptr {{%.+}}, metadata !"[[TYPE1B:_ZTSFu3i32S_E]]")
    a(1);
}

pub fn bar1() {
    foo1(|a| -> i32 { a + 1 });
    // CHECK-LABEL: define{{.*}}4bar1{{.*}}$u7b$$u7b$closure$u7d$$u7d
    // CHECK-SAME:  {{.*}}!type ![[TYPE1A:[[:print:]]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
    fn a(a: i32) -> i32 { a + 1 }
    foo1(a);
    // CHECK-LABEL: define{{.*}}4bar11a
    // CHECK-SAME:  {{.*}}!type ![[TYPE1A:[[:print:]]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
}

pub fn foo2(a: &dyn Fn(i32) -> i32) {
    // CHECK-LABEL: define{{.*}}4foo2{{.*}}!type !{{[0-9]+}}
    // CHECK:       call i1 @llvm.type.test(ptr {{%.+}}, metadata !"[[TYPE1B]]")
    a(1);
}

pub fn bar2() {
    foo2(&|a: i32| -> i32 { a + 1 });
    // CHECK-LABEL: define{{.*}}4bar2{{.*}}$u7b$$u7b$closure$u7d$$u7d
    // CHECK-SAME:  {{.*}}!type ![[TYPE1A:[[:print:]]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
    fn a(a: i32) -> i32 { a + 1 }
    foo2(&a);
    // CHECK-LABEL: define{{.*}}4bar21a
    // CHECK-SAME:  {{.*}}!type ![[TYPE1A:[[:print:]]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
}

pub fn foo3(a: &mut dyn FnMut(i32) -> i32) {
    // CHECK-LABEL: define{{.*}}4foo3{{.*}}!type !{{[0-9]+}}
    // CHECK:       call i1 @llvm.type.test(ptr {{%.+}}, metadata !"[[TYPE1B]]")
    a(1);
}

pub fn bar3() {
    foo3(&mut |a: i32| -> i32 { a + 1 });
    // CHECK-LABEL: define{{.*}}4bar3{{.*}}$u7b$$u7b$closure$u7d$$u7d
    // CHECK-SAME:  {{.*}}!type ![[TYPE1A:[[:print:]]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
    fn a(a: i32) -> i32 { a + 1 }
    foo3(&mut a);
    // CHECK-LABEL: define{{.*}}4bar31a
    // CHECK-SAME:  {{.*}}!type ![[TYPE1A:[[:print:]]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
}

// CHECK: ![[TYPE1A]] = !{i64 0, !"[[TYPE1B]]"}
