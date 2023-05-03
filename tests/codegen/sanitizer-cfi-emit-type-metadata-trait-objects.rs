// Verifies that type metadata identifiers for trait objects are emitted correctly.
//
// needs-sanitizer-cfi
// compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi

#![crate_type="lib"]

trait Trait1 {
    fn foo(&self);
}

struct Type1;

impl Trait1 for Type1 {
    fn foo(&self) {
    }
}

pub fn foo() {
    let a = Type1;
    a.foo();
    // CHECK-LABEL: define{{.*}}foo{{.*}}!type !{{[0-9]+}}
    // CHECK:       call <sanitizer_cfi_emit_type_metadata_trait_objects::Type1 as sanitizer_cfi_emit_type_metadata_trait_objects::Trait1>::foo
}

pub fn bar() {
    let a = Type1;
    let b = &a as &dyn Trait1;
    b.foo();
    // CHECK-LABEL: define{{.*}}bar{{.*}}!type !{{[0-9]+}}
    // CHECK:       call i1 @llvm.type.test({{i8\*|ptr}} {{%f|%0|%1}}, metadata !"[[TYPE1:[[:print:]]+]]")
}

pub fn baz() {
    let a = Type1;
    let b = &a as &dyn Trait1;
    a.foo();
    b.foo();
    // CHECK-LABEL: define{{.*}}baz{{.*}}!type !{{[0-9]+}}
    // CHECK:       call <sanitizer_cfi_emit_type_metadata_trait_objects::Type1 as sanitizer_cfi_emit_type_metadata_trait_objects::Trait1>::foo
    // CHECK:       call i1 @llvm.type.test({{i8\*|ptr}} {{%f|%0|%1}}, metadata !"[[TYPE1:[[:print:]]+]]")
}

// CHECK: !{{[0-9]+}} = !{i64 0, !"[[TYPE1]]"}
