// Verifies that type metadata identifiers for functions are emitted correctly
// for paths.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Copt-level=0 -Zsanitizer=cfi -Ctarget-feature=-crt-static -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]
#![feature(type_alias_impl_trait)]

extern crate core;

pub type Type1 = impl Send;
pub type Type2 = impl Send;
pub type Type3 = impl Send;
pub type Type4 = impl Send;

#[define_opaque(Type1, Type2, Type4)]
pub fn foo() {
    // Type in extern path
    extern "C" {
        fn bar();
    }
    let _: Type1 = bar;

    // Type in closure path
    || {
        pub struct Foo;
        let _: Type2 = Foo;
    };

    // Type in const path
    const {
        pub struct Foo;
        #[define_opaque(Type3)]
        fn bar() -> Type3 {
            Foo
        }
    };

    // Type in impl path
    struct Foo;
    impl Foo {
        fn bar(&self) {}
    }
    let _: Type4 = <Foo>::bar;
}

// Force arguments to be passed by using a reference. Otherwise, they may end up PassMode::Ignore

pub fn foo1(_: &Type1) {}
// CHECK: define{{.*}}4foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo2(_: &Type1, _: &Type1) {}
// CHECK: define{{.*}}4foo2{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo3(_: &Type1, _: &Type1, _: &Type1) {}
// CHECK: define{{.*}}4foo3{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo4(_: &Type2) {}
// CHECK: define{{.*}}4foo4{{.*}}!type ![[TYPE4:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo5(_: &Type2, _: &Type2) {}
// CHECK: define{{.*}}4foo5{{.*}}!type ![[TYPE5:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo6(_: &Type2, _: &Type2, _: &Type2) {}
// CHECK: define{{.*}}4foo6{{.*}}!type ![[TYPE6:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo7(_: &Type3) {}
// CHECK: define{{.*}}4foo7{{.*}}!type ![[TYPE7:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo8(_: &Type3, _: &Type3) {}
// CHECK: define{{.*}}4foo8{{.*}}!type ![[TYPE8:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo9(_: &Type3, _: &Type3, _: &Type3) {}
// CHECK: define{{.*}}4foo9{{.*}}!type ![[TYPE9:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo10(_: &Type4) {}
// CHECK: define{{.*}}5foo10{{.*}}!type ![[TYPE10:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo11(_: &Type4, _: &Type4) {}
// CHECK: define{{.*}}5foo11{{.*}}!type ![[TYPE11:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo12(_: &Type4, _: &Type4, _: &Type4) {}
// CHECK: define{{.*}}5foo12{{.*}}!type ![[TYPE12:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NvNFNvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo10{{[{}][{}]}}extern{{[}][}]}}3barEE"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NvNFNvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo10{{[{}][{}]}}extern{{[}][}]}}3barES0_E"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NvNFNvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo10{{[{}][{}]}}extern{{[}][}]}}3barES0_S0_E"}
// CHECK: ![[TYPE4]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtNCNvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo11{{[{}][{}]}}closure{{[}][}]}}3FooEE"}
// CHECK: ![[TYPE5]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtNCNvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo11{{[{}][{}]}}closure{{[}][}]}}3FooES0_E"}
// CHECK: ![[TYPE6]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtNCNvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo11{{[{}][{}]}}closure{{[}][}]}}3FooES0_S0_E"}
// CHECK: ![[TYPE7]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtNkNvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo12{{[{}][{}]}}constant{{[}][}]}}3FooEE"}
// CHECK: ![[TYPE8]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtNkNvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo12{{[{}][{}]}}constant{{[}][}]}}3FooES0_E"}
// CHECK: ![[TYPE9]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtNkNvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo12{{[{}][{}]}}constant{{[}][}]}}3FooES0_S0_E"}
// CHECK: ![[TYPE10]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NvNINvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo8{{[{}][{}]}}impl{{[}][}]}}3barEE"}
// CHECK: ![[TYPE11]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NvNINvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo8{{[{}][{}]}}impl{{[}][}]}}3barES0_E"}
// CHECK: ![[TYPE12]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NvNINvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo8{{[{}][{}]}}impl{{[}][}]}}3barES0_S0_E"}
