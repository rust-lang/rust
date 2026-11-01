// Verifies that type metadata identifiers for functions are emitted correctly
// for trait types.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Copt-level=0 -Zsanitizer=cfi -Ctarget-feature=-crt-static -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]

extern crate core;

pub trait Trait1 {
    fn foo(&self);
}

#[derive(Clone, Copy)]
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

#[derive(Copy, Clone)]
pub struct Type5;

impl<T, U, const N: usize> Trait5<U, N> for T {
    fn quux(&self, _: &[U; N]) {}
}

pub fn foo1(_: &dyn Send) {}
// CHECK: define{{.*}}4foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo2(_: &dyn Send, _: &dyn Send) {}
// CHECK: define{{.*}}4foo2{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo3(_: &dyn Send, _: &dyn Send, _: &dyn Send) {}
// CHECK: define{{.*}}4foo3{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo4(_: &(dyn Send + Sync)) {}
// CHECK: define{{.*}}4foo4{{.*}}!type ![[TYPE4:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo5(_: &(dyn Send + Sync), _: &(dyn Sync + Send)) {}
// CHECK: define{{.*}}4foo5{{.*}}!type ![[TYPE5:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo6(_: &(dyn Send + Sync), _: &(dyn Sync + Send), _: &(dyn Sync + Send)) {}
// CHECK: define{{.*}}4foo6{{.*}}!type ![[TYPE6:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo7(_: &(dyn Trait1 + Send)) {}
// CHECK: define{{.*}}4foo7{{.*}}!type ![[TYPE7:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo8(_: &(dyn Trait1 + Send), _: &(dyn Trait1 + Send)) {}
// CHECK: define{{.*}}4foo8{{.*}}!type ![[TYPE8:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo9(_: &(dyn Trait1 + Send), _: &(dyn Trait1 + Send), _: &(dyn Trait1 + Send)) {}
// CHECK: define{{.*}}4foo9{{.*}}!type ![[TYPE9:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo10(_: &(dyn Trait1 + Send + Sync)) {}
// CHECK: define{{.*}}5foo10{{.*}}!type ![[TYPE10:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo11(_: &(dyn Trait1 + Send + Sync), _: &(dyn Trait1 + Sync + Send)) {}
// CHECK: define{{.*}}5foo11{{.*}}!type ![[TYPE11:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo12(
    _: &(dyn Trait1 + Send + Sync),
    _: &(dyn Trait1 + Sync + Send),
    _: &(dyn Trait1 + Sync + Send),
) {
}
// CHECK: define{{.*}}5foo12{{.*}}!type ![[TYPE12:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo13(_: &dyn Trait1) {}
// CHECK: define{{.*}}5foo13{{.*}}!type ![[TYPE13:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo14(_: &dyn Trait1, _: &dyn Trait1) {}
// CHECK: define{{.*}}5foo14{{.*}}!type ![[TYPE14:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo15(_: &dyn Trait1, _: &dyn Trait1, _: &dyn Trait1) {}
// CHECK: define{{.*}}5foo15{{.*}}!type ![[TYPE15:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo16<T>(_: &dyn Trait2<T>) {}
pub fn bar16() {
    let a = Type2;
    foo16(&a);
}
// CHECK: define{{.*}}5foo16{{.*}}!type ![[TYPE16:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo17<T>(_: &dyn Trait2<T>, _: &dyn Trait2<T>) {}
pub fn bar17() {
    let a = Type2;
    foo17(&a, &a);
}
// CHECK: define{{.*}}5foo17{{.*}}!type ![[TYPE17:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo18<T>(_: &dyn Trait2<T>, _: &dyn Trait2<T>, _: &dyn Trait2<T>) {}
pub fn bar18() {
    let a = Type2;
    foo18(&a, &a, &a);
}
// CHECK: define{{.*}}5foo18{{.*}}!type ![[TYPE18:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo19(_: &dyn Trait3<Type3>) {}
// CHECK: define{{.*}}5foo19{{.*}}!type ![[TYPE19:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo20(_: &dyn Trait3<Type3>, _: &dyn Trait3<Type3>) {}
// CHECK: define{{.*}}5foo20{{.*}}!type ![[TYPE20:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo21(_: &dyn Trait3<Type3>, _: &dyn Trait3<Type3>, _: &dyn Trait3<Type3>) {}
// CHECK: define{{.*}}5foo21{{.*}}!type ![[TYPE21:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo22<'a>(_: &dyn Trait4<'a, Type4, Output = &'a i32>) {}
// CHECK: define{{.*}}5foo22{{.*}}!type ![[TYPE22:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo23<'a>(
    _: &dyn Trait4<'a, Type4, Output = &'a i32>,
    _: &dyn Trait4<'a, Type4, Output = &'a i32>,
) {
}
// CHECK: define{{.*}}5foo23{{.*}}!type ![[TYPE23:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo24<'a>(
    _: &dyn Trait4<'a, Type4, Output = &'a i32>,
    _: &dyn Trait4<'a, Type4, Output = &'a i32>,
    _: &dyn Trait4<'a, Type4, Output = &'a i32>,
) {
}
// CHECK: define{{.*}}5foo24{{.*}}!type ![[TYPE24:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo25(_: &dyn Trait5<Type5, 32>) {}
// CHECK: define{{.*}}5foo25{{.*}}!type ![[TYPE25:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo26(_: &dyn Trait5<Type5, 32>, _: &dyn Trait5<Type5, 32>) {}
// CHECK: define{{.*}}5foo26{{.*}}!type ![[TYPE26:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo27(_: &dyn Trait5<Type5, 32>, _: &dyn Trait5<Type5, 32>, _: &dyn Trait5<Type5, 32>) {}
// CHECK: define{{.*}}5foo27{{.*}}!type ![[TYPE27:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

// CHECK: ![[TYPE13]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait1u6regionEEE"}
// CHECK: ![[TYPE16]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait2Iu3i32Eu6regionEEE"}
// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker4Sendu6regionEEE"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker4Sendu6regionEES2_E"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker4Sendu6regionEES2_S2_E"}
// FIXME(rcvalle): Enforce autotraits ordering when encoding (e.g., alphabetical order)
// CHECK: ![[TYPE4]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker{{(4Send|4Sync)}}u{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker{{(4Send|4Sync)}}u6regionEEE"}
// CHECK: ![[TYPE5]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker{{(4Send|4Sync)}}u{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker{{(4Send|4Sync)}}u6regionEES3_E"}
// CHECK: ![[TYPE6]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker{{(4Send|4Sync)}}u{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker{{(4Send|4Sync)}}u6regionEES3_S3_E"}
// CHECK: ![[TYPE7]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait1u{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker4Sendu6regionEEE"}
// CHECK: ![[TYPE8]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait1u{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker4Sendu6regionEES3_E"}
// CHECK: ![[TYPE9]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait1u{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker4Sendu6regionEES3_S3_E"}
// CHECK: ![[TYPE10]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait1u{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker{{(4Send|4Sync)}}u{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker{{(4Send|4Sync)}}u6regionEEE"}
// CHECK: ![[TYPE11]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait1u{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker{{(4Send|4Sync)}}u{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker{{(4Send|4Sync)}}u6regionEES4_E"}
// CHECK: ![[TYPE12]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait1u{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker{{(4Send|4Sync)}}u{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker{{(4Send|4Sync)}}u6regionEES4_S4_E"}
// CHECK: ![[TYPE14]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait1u6regionEES2_E"}
// CHECK: ![[TYPE15]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait1u6regionEES2_S2_E"}
// CHECK: ![[TYPE17]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait2Iu3i32Eu6regionEES3_E"}
// CHECK: ![[TYPE18]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait2Iu3i32Eu6regionEES3_S3_E"}
// CHECK: ![[TYPE19]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait3Iu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}5Type3Eu6regionEEE"}
// CHECK: ![[TYPE20]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait3Iu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}5Type3Eu6regionEES3_E"}
// CHECK: ![[TYPE21]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait3Iu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}5Type3Eu6regionEES3_S3_E"}
// CHECK: ![[TYPE22]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait4Iu6regionu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}5Type4Eu{{[0-9]+}}NtNtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait46OutputIS_S0_Eu3refIu3i32ES_EEE"}
// CHECK: ![[TYPE23]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait4Iu6regionu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}5Type4Eu{{[0-9]+}}NtNtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait46OutputIS_S0_Eu3refIu3i32ES_EES6_E"}
// CHECK: ![[TYPE24]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait4Iu6regionu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}5Type4Eu{{[0-9]+}}NtNtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait46OutputIS_S0_Eu3refIu3i32ES_EES6_S6_E"}
// CHECK: ![[TYPE25]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait5Iu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}5Type5Lu5usize32EEu6regionEEE"}
// CHECK: ![[TYPE26]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait5Iu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}5Type5Lu5usize32EEu6regionEES5_E"}
// CHECK: ![[TYPE27]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Trait5Iu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}5Type5Lu5usize32EEu6regionEES5_S5_E"}
