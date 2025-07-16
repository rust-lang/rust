// Verifies that type metadata identifiers for trait objects are emitted correctly.
//
//@ add-core-stubs
//@ revisions: aarch64 x86_64
//@ [aarch64] compile-flags: --target aarch64-unknown-none
//@ [aarch64] needs-llvm-components: aarch64
//@ [x86_64] compile-flags: --target x86_64-unknown-none
//@ [x86_64] needs-llvm-components: x86
//@ compile-flags: -Cno-prepopulate-passes -Zsanitizer=kcfi -Copt-level=0

#![crate_type = "lib"]
#![feature(arbitrary_self_types, no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

pub trait Trait1 {
    fn foo(&self);
}

pub struct Type1;

impl Trait1 for Type1 {
    fn foo(&self) {}
}

pub trait Trait2<T> {
    fn bar(&self);
}

pub struct Type2;

impl Trait2<i32> for Type2 {
    fn bar(&self) {}
}

pub trait Trait3<T> {
    fn baz(&self, _: &T);
}

pub struct Type3;

impl<T, U> Trait3<U> for T {
    fn baz(&self, _: &U) {}
}

pub trait Trait4<'a, T> {
    type Output: 'a;
    fn qux(&self, _: &T) -> Self::Output;
}

pub struct Type4;

impl<'a, T, U> Trait4<'a, U> for T {
    type Output = &'a i32;
    fn qux(&self, _: &U) -> Self::Output {
        &0
    }
}

pub trait Trait5<T, const N: usize> {
    fn quux(&self, _: &[T; N]);
}

pub struct Type5;

impl Copy for Type5 {}

impl<T, U, const N: usize> Trait5<U, N> for T {
    fn quux(&self, _: &[U; N]) {}
}

pub fn foo1(a: &dyn Trait1) {
    a.foo();
    // CHECK-LABEL: define{{.*}}4foo1{{.*}}!{{<unknown kind #36>|kcfi_type}} !{{[0-9]+}}
    // CHECK:       call void %{{[0-9]}}(ptr align 1 {{%[a-z]\.0|%_[0-9]}}){{.*}}[ "kcfi"(i32 [[TYPE1:[[:print:]]+]]) ]
}

pub fn bar1() {
    let a = Type1;
    let b = &a as &dyn Trait1;
    b.foo();
    // CHECK-LABEL: define{{.*}}4bar1{{.*}}!{{<unknown kind #36>|kcfi_type}} !{{[0-9]+}}
    // CHECK:       call void %{{[0-9]}}(ptr align 1 {{%[a-z]\.0|%_[0-9]}}){{.*}}[ "kcfi"(i32 [[TYPE1:[[:print:]]+]]) ]
}

pub fn foo2<T>(a: &dyn Trait2<T>) {
    a.bar();
    // CHECK-LABEL: define{{.*}}4foo2{{.*}}!{{<unknown kind #36>|kcfi_type}} !{{[0-9]+}}
    // CHECK:       call void %{{[0-9]}}(ptr align 1 {{%[a-z]\.0|%_[0-9]}}){{.*}}[ "kcfi"(i32 [[TYPE2:[[:print:]]+]]) ]
}

pub fn bar2() {
    let a = Type2;
    foo2(&a);
    let b = &a as &dyn Trait2<i32>;
    b.bar();
    // CHECK-LABEL: define{{.*}}4bar2{{.*}}!{{<unknown kind #36>|kcfi_type}} !{{[0-9]+}}
    // CHECK:       call void %{{[0-9]}}(ptr align 1 {{%[a-z]\.0|%_[0-9]}}){{.*}}[ "kcfi"(i32 [[TYPE2:[[:print:]]+]]) ]
}

pub fn foo3(a: &dyn Trait3<Type3>) {
    let b = Type3;
    a.baz(&b);
    // CHECK-LABEL: define{{.*}}4foo3{{.*}}!{{<unknown kind #36>|kcfi_type}} !{{[0-9]+}}
    // CHECK:       call void %{{[0-9]}}(ptr align 1 {{%[a-z]\.0|%_[0-9]}}, ptr align 1 {{%[a-z]\.0|%_[0-9]}}){{.*}}[ "kcfi"(i32 [[TYPE3:[[:print:]]+]]) ]
}

pub fn bar3() {
    let a = Type3;
    foo3(&a);
    let b = &a as &dyn Trait3<Type3>;
    b.baz(&a);
    // CHECK-LABEL: define{{.*}}4bar3{{.*}}!{{<unknown kind #36>|kcfi_type}} !{{[0-9]+}}
    // CHECK:       call void %{{[0-9]}}(ptr align 1 {{%[a-z]\.0|%_[0-9]}}, ptr align 1 {{%[a-z]\.0|%_[0-9]}}){{.*}}[ "kcfi"(i32 [[TYPE3:[[:print:]]+]]) ]
}

pub fn foo4<'a>(a: &dyn Trait4<'a, Type4, Output = &'a i32>) {
    let b = Type4;
    a.qux(&b);
    // CHECK-LABEL: define{{.*}}4foo4{{.*}}!{{<unknown kind #36>|kcfi_type}} !{{[0-9]+}}
    // CHECK:       call align 4 ptr %{{[0-9]}}(ptr align 1 {{%[a-z]\.0|%_[0-9]}}, ptr align 1 {{%[a-z]\.0|%_[0-9]}}){{.*}}[ "kcfi"(i32 [[TYPE4:[[:print:]]+]]) ]
}

pub fn bar4<'a>() {
    let a = Type4;
    foo4(&a);
    let b = &a as &dyn Trait4<'a, Type4, Output = &'a i32>;
    b.qux(&a);
    // CHECK-LABEL: define{{.*}}4bar4{{.*}}!{{<unknown kind #36>|kcfi_type}} !{{[0-9]+}}
    // CHECK:       call align 4 ptr %{{[0-9]}}(ptr align 1 {{%[a-z]\.0|%_[0-9]}}, ptr align 1 {{%[a-z]\.0|%_[0-9]}}){{.*}}[ "kcfi"(i32 [[TYPE4:[[:print:]]+]]) ]
}

pub fn foo5(a: &dyn Trait5<Type5, 32>) {
    let b = &[Type5; 32];
    a.quux(&b);
    // CHECK-LABEL: define{{.*}}4foo5{{.*}}!{{<unknown kind #36>|kcfi_type}} !{{[0-9]+}}
    // CHECK:       call void %{{[0-9]}}(ptr align 1 {{%[a-z](\.0)*|%_[0-9]+]}}, ptr align 1 {{%[a-z](\.0)*|%_[0-9]+}}){{.*}}[ "kcfi"(i32 [[TYPE5:[[:print:]]+]]) ]
}

pub fn bar5() {
    let a = &[Type5; 32];
    foo5(&a);
    let b = &a as &dyn Trait5<Type5, 32>;
    b.quux(&a);
    // CHECK-LABEL: define{{.*}}4bar5{{.*}}!{{<unknown kind #36>|kcfi_type}} !{{[0-9]+}}
    // CHECK:       call void %{{[0-9]}}(ptr align 1 {{%[a-z](\.0)*|%_[0-9]+]}}, ptr align 1 {{%[a-z](\.0)*|%_[0-9]+}}){{.*}}[ "kcfi"(i32 [[TYPE5:[[:print:]]+]]) ]
}

// CHECK: !{{[0-9]+}} = !{i32 [[TYPE1]]}
// CHECK: !{{[0-9]+}} = !{i32 [[TYPE2]]}
// CHECK: !{{[0-9]+}} = !{i32 [[TYPE3]]}
// CHECK: !{{[0-9]+}} = !{i32 [[TYPE4]]}
// CHECK: !{{[0-9]+}} = !{i32 [[TYPE5]]}
