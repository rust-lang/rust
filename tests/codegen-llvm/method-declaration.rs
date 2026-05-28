//@ compile-flags: -g -Cno-prepopulate-passes

// Verify that we added a declaration for a method.

// CHECK: define{{.*}}@method{{.*}} !dbg ![[METHOD_DEF_DBG:[0-9]+]]
// CHECK: define{{.*}}@function{{.*}} !dbg ![[FUNC_DEF_DBG:[0-9]+]]

#![crate_type = "lib"]

// CHECK-DAG: ![[FOO_DBG:[0-9]+]] = !DICompositeType(tag: {{.*}} name: "Foo", {{.*}} identifier:
pub struct Foo;

impl Foo {
    // CHECK-DAG: ![[METHOD_DEF_DBG]] = distinct !DISubprogram(name: "method"{{.*}}, scope: ![[FOO_DBG]]{{.*}}DISPFlagDefinition{{.*}}, declaration: ![[METHOD_DECL_DBG:[0-9]+]]
    // CHECK-DAG: ![[METHOD_DECL_DBG]] = !DISubprogram(name: "method"{{.*}}, scope: ![[FOO_DBG]]
    #[no_mangle]
    pub fn method() {}
}

// CHECK: ![[FUNC_DEF_DBG]] = distinct !DISubprogram(name: "function"
// CHECK-NOT: declaration
// CHECK-SAME: DISPFlagDefinition
// CHECK-NOT: declaration
// CHECK-SAME: )
#[no_mangle]
pub fn function() {}
