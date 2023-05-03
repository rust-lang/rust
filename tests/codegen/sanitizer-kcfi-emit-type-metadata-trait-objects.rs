// Verifies that type metadata identifiers for trait objects are emitted correctly.
//
// revisions: aarch64 x86_64
// [aarch64] compile-flags: --target aarch64-unknown-none
// [aarch64] needs-llvm-components: aarch64
// [x86_64] compile-flags: --target x86_64-unknown-none
// [x86_64] needs-llvm-components:
// compile-flags: -Cno-prepopulate-passes -Zsanitizer=kcfi -Copt-level=0

#![crate_type="lib"]
#![feature(arbitrary_self_types, no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized { }
#[lang="copy"]
trait Copy { }
#[lang="receiver"]
trait Receiver { }
#[lang="dispatch_from_dyn"]
trait DispatchFromDyn<T> { }
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<&'a U> for &'a T {}
#[lang = "unsize"]
trait Unsize<T: ?Sized> { }
#[lang = "coerce_unsized"]
pub trait CoerceUnsized<T: ?Sized> { }
impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}
#[lang="freeze"]
trait Freeze { }
#[lang="drop_in_place"]
fn drop_in_place_fn<T>() { }

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
    // CHECK-LABEL: define{{.*}}foo{{.*}}!{{<unknown kind #36>|kcfi_type}} !{{[0-9]+}}
    // CHECK:       call <sanitizer_kcfi_emit_type_metadata_trait_objects::Type1 as sanitizer_kcfi_emit_type_metadata_trait_objects::Trait1>::foo
}

pub fn bar() {
    let a = Type1;
    let b = &a as &dyn Trait1;
    b.foo();
    // CHECK-LABEL: define{{.*}}bar{{.*}}!{{<unknown kind #36>|kcfi_type}} !{{[0-9]+}}
    // CHECK:       call void %0({{\{\}\*|ptr}} align 1 {{%b\.0|%_1}}){{.*}}[ "kcfi"(i32 [[TYPE1:[[:print:]]+]]) ]
}

pub fn baz() {
    let a = Type1;
    let b = &a as &dyn Trait1;
    a.foo();
    b.foo();
    // CHECK-LABEL: define{{.*}}baz{{.*}}!{{<unknown kind #36>|kcfi_type}} !{{[0-9]+}}
    // CHECK:       call <sanitizer_kcfi_emit_type_metadata_trait_objects::Type1 as sanitizer_kcfi_emit_type_metadata_trait_objects::Trait1>::foo
    // CHECK:       call void %0({{\{\}\*|ptr}} align 1 {{%b\.0|%_1}}){{.*}}[ "kcfi"(i32 [[TYPE1:[[:print:]]+]]) ]
}

// CHECK: !{{[0-9]+}} = !{i32 [[TYPE1]]}
