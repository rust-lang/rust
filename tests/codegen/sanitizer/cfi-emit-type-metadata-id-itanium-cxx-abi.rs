// Verifies that type metadata identifiers for functions are emitted correctly.
//
// needs-sanitizer-cfi
// compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi

#![crate_type="lib"]
#![allow(dead_code)]
#![allow(incomplete_features)]
#![allow(unused_must_use)]
#![feature(adt_const_params, extern_types, inline_const, type_alias_impl_trait)]

extern crate core;
use core::ffi::*;
use std::marker::PhantomData;

// User-defined type (structure)
pub struct Struct1<T> {
    member1: T,
}

// User-defined type (enum)
pub enum Enum1<T> {
    Variant1(T),
}

// User-defined type (union)
pub union Union1<T> {
    member1: std::mem::ManuallyDrop<T>,
}

// Extern type
extern {
    pub type type1;
}

// Trait
pub trait Trait1<T> {
    fn foo(&self) { }
}

// Trait implementation
impl<T> Trait1<T> for i32 {
    fn foo(&self) { }
}

// Trait implementation
impl<T, U> Trait1<T> for Struct1<U> {
    fn foo(&self) { }
}

// impl Trait type aliases for helping with defining other types (see below)
pub type Type1 = impl Send;
pub type Type2 = impl Send;
pub type Type3 = impl Send;
pub type Type4 = impl Send;
pub type Type5 = impl Send;
pub type Type6 = impl Send;
pub type Type7 = impl Send;
pub type Type8 = impl Send;
pub type Type9 = impl Send;
pub type Type10 = impl Send;
pub type Type11 = impl Send;

pub fn fn1<'a>() where
    Type1: 'static,
    Type2: 'static,
    Type3: 'static,
    Type4: 'static,
    Type5: 'static,
    Type6: 'static,
    Type7: 'static,
    Type8: 'static,
    Type9: 'static,
    Type10: 'static,
    Type11: 'static,
{
    // Closure
    let closure1 = || { };
    let _: Type1 = closure1;

    // Constructor
    pub struct Foo(i32);
    let _: Type2 = Foo;

    // Type in extern path
    extern {
        fn foo();
    }
    let _: Type3 = foo;

    // Type in closure path
    || {
        pub struct Foo;
        let _: Type4 = Foo;
    };

    // Type in const path
    const {
        pub struct Foo;
        fn foo() -> Type5 { Foo }
    };

    // Type in impl path
    impl<T> Struct1<T> {
        fn foo(&self) { }
    }
    let _: Type6 = <Struct1<i32>>::foo;

    // Trait method
    let _: Type7 = <dyn Trait1<i32>>::foo;

    // Trait method
    let _: Type8 = <i32 as Trait1<i32>>::foo;

    // Trait method
    let _: Type9 = <Struct1<i32> as Trait1<i32>>::foo;

    // Const generics
    pub struct Qux<T, const N: usize>([T; N]);
    let _: Type10 = Qux([0; 32]);

    // Lifetimes/regions
    pub struct Quux<'a>(&'a i32);
    pub struct Quuux<'a, 'b>(&'a i32, &'b Quux<'b>);
    let _: Type11 = Quuux;
}

// Helper type to make Type12 have an unique id
struct Foo(i32);

// repr(transparent) user-defined type
#[repr(transparent)]
pub struct Type12 {
    member1: (),
    member2: PhantomData<i32>,
    member3: Foo,
}

// Self-referencing repr(transparent) user-defined type
#[repr(transparent)]
pub struct Type13<'a> {
    member1: (),
    member2: PhantomData<i32>,
    member3: &'a Type13<'a>,
}

// Helper type to make Type14 have an unique id
pub struct Bar;

// repr(transparent) user-defined generic type
#[repr(transparent)]
pub struct Type14<T>(T);

pub fn foo0(_: ()) { }
// CHECK: define{{.*}}foo0{{.*}}!type ![[TYPE0:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo1(_: (), _: c_void) { }
// CHECK: define{{.*}}foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo2(_: (), _: c_void, _: c_void) { }
// CHECK: define{{.*}}foo2{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo3(_: *mut ()) { }
// CHECK: define{{.*}}foo3{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo4(_: *mut (), _: *mut c_void) { }
// CHECK: define{{.*}}foo4{{.*}}!type ![[TYPE4:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo5(_: *mut (), _: *mut c_void, _: *mut c_void) { }
// CHECK: define{{.*}}foo5{{.*}}!type ![[TYPE5:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo6(_: *const ()) { }
// CHECK: define{{.*}}foo6{{.*}}!type ![[TYPE6:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo7(_: *const (), _: *const c_void) { }
// CHECK: define{{.*}}foo7{{.*}}!type ![[TYPE7:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo8(_: *const (), _: *const c_void, _: *const c_void) { }
// CHECK: define{{.*}}foo8{{.*}}!type ![[TYPE8:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo9(_: bool) { }
// CHECK: define{{.*}}foo9{{.*}}!type ![[TYPE9:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo10(_: bool, _: bool) { }
// CHECK: define{{.*}}foo10{{.*}}!type ![[TYPE10:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo11(_: bool, _: bool, _: bool) { }
// CHECK: define{{.*}}foo11{{.*}}!type ![[TYPE11:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo12(_: i8) { }
// CHECK: define{{.*}}foo12{{.*}}!type ![[TYPE12:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo13(_: i8, _: i8) { }
// CHECK: define{{.*}}foo13{{.*}}!type ![[TYPE13:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo14(_: i8, _: i8, _: i8) { }
// CHECK: define{{.*}}foo14{{.*}}!type ![[TYPE14:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo15(_: i16) { }
// CHECK: define{{.*}}foo15{{.*}}!type ![[TYPE15:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo16(_: i16, _: i16) { }
// CHECK: define{{.*}}foo16{{.*}}!type ![[TYPE16:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo17(_: i16, _: i16, _: i16) { }
// CHECK: define{{.*}}foo17{{.*}}!type ![[TYPE17:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo18(_: i32) { }
// CHECK: define{{.*}}foo18{{.*}}!type ![[TYPE18:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo19(_: i32, _: i32) { }
// CHECK: define{{.*}}foo19{{.*}}!type ![[TYPE19:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo20(_: i32, _: i32, _: i32) { }
// CHECK: define{{.*}}foo20{{.*}}!type ![[TYPE20:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo21(_: i64) { }
// CHECK: define{{.*}}foo21{{.*}}!type ![[TYPE21:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo22(_: i64, _: i64) { }
// CHECK: define{{.*}}foo22{{.*}}!type ![[TYPE22:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo23(_: i64, _: i64, _: i64) { }
// CHECK: define{{.*}}foo23{{.*}}!type ![[TYPE23:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo24(_: i128) { }
// CHECK: define{{.*}}foo24{{.*}}!type ![[TYPE24:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo25(_: i128, _: i128) { }
// CHECK: define{{.*}}foo25{{.*}}!type ![[TYPE25:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo26(_: i128, _: i128, _: i128) { }
// CHECK: define{{.*}}foo26{{.*}}!type ![[TYPE26:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo27(_: isize) { }
// CHECK: define{{.*}}foo27{{.*}}!type ![[TYPE27:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo28(_: isize, _: isize) { }
// CHECK: define{{.*}}foo28{{.*}}!type ![[TYPE28:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo29(_: isize, _: isize, _: isize) { }
// CHECK: define{{.*}}foo29{{.*}}!type ![[TYPE29:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo30(_: u8) { }
// CHECK: define{{.*}}foo30{{.*}}!type ![[TYPE30:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo31(_: u8, _: u8) { }
// CHECK: define{{.*}}foo31{{.*}}!type ![[TYPE31:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo32(_: u8, _: u8, _: u8) { }
// CHECK: define{{.*}}foo32{{.*}}!type ![[TYPE32:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo33(_: u16) { }
// CHECK: define{{.*}}foo33{{.*}}!type ![[TYPE33:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo34(_: u16, _: u16) { }
// CHECK: define{{.*}}foo34{{.*}}!type ![[TYPE34:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo35(_: u16, _: u16, _: u16) { }
// CHECK: define{{.*}}foo35{{.*}}!type ![[TYPE35:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo36(_: u32) { }
// CHECK: define{{.*}}foo36{{.*}}!type ![[TYPE36:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo37(_: u32, _: u32) { }
// CHECK: define{{.*}}foo37{{.*}}!type ![[TYPE37:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo38(_: u32, _: u32, _: u32) { }
// CHECK: define{{.*}}foo38{{.*}}!type ![[TYPE38:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo39(_: u64) { }
// CHECK: define{{.*}}foo39{{.*}}!type ![[TYPE39:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo40(_: u64, _: u64) { }
// CHECK: define{{.*}}foo40{{.*}}!type ![[TYPE40:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo41(_: u64, _: u64, _: u64) { }
// CHECK: define{{.*}}foo41{{.*}}!type ![[TYPE41:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo42(_: u128) { }
// CHECK: define{{.*}}foo42{{.*}}!type ![[TYPE42:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo43(_: u128, _: u128) { }
// CHECK: define{{.*}}foo43{{.*}}!type ![[TYPE43:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo44(_: u128, _: u128, _: u128) { }
// CHECK: define{{.*}}foo44{{.*}}!type ![[TYPE44:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo45(_: usize) { }
// CHECK: define{{.*}}foo45{{.*}}!type ![[TYPE45:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo46(_: usize, _: usize) { }
// CHECK: define{{.*}}foo46{{.*}}!type ![[TYPE46:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo47(_: usize, _: usize, _: usize) { }
// CHECK: define{{.*}}foo47{{.*}}!type ![[TYPE47:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo48(_: f32) { }
// CHECK: define{{.*}}foo48{{.*}}!type ![[TYPE48:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo49(_: f32, _: f32) { }
// CHECK: define{{.*}}foo49{{.*}}!type ![[TYPE49:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo50(_: f32, _: f32, _: f32) { }
// CHECK: define{{.*}}foo50{{.*}}!type ![[TYPE50:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo51(_: f64) { }
// CHECK: define{{.*}}foo51{{.*}}!type ![[TYPE51:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo52(_: f64, _: f64) { }
// CHECK: define{{.*}}foo52{{.*}}!type ![[TYPE52:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo53(_: f64, _: f64, _: f64) { }
// CHECK: define{{.*}}foo53{{.*}}!type ![[TYPE53:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo54(_: char) { }
// CHECK: define{{.*}}foo54{{.*}}!type ![[TYPE54:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo55(_: char, _: char) { }
// CHECK: define{{.*}}foo55{{.*}}!type ![[TYPE55:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo56(_: char, _: char, _: char) { }
// CHECK: define{{.*}}foo56{{.*}}!type ![[TYPE56:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo57(_: &str) { }
// CHECK: define{{.*}}foo57{{.*}}!type ![[TYPE57:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo58(_: &str, _: &str) { }
// CHECK: define{{.*}}foo58{{.*}}!type ![[TYPE58:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo59(_: &str, _: &str, _: &str) { }
// CHECK: define{{.*}}foo59{{.*}}!type ![[TYPE59:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo60(_: (i32, i32)) { }
// CHECK: define{{.*}}foo60{{.*}}!type ![[TYPE60:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo61(_: (i32, i32), _: (i32, i32)) { }
// CHECK: define{{.*}}foo61{{.*}}!type ![[TYPE61:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo62(_: (i32, i32), _: (i32, i32), _: (i32, i32)) { }
// CHECK: define{{.*}}foo62{{.*}}!type ![[TYPE62:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo63(_: [i32; 32]) { }
// CHECK: define{{.*}}foo63{{.*}}!type ![[TYPE63:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo64(_: [i32; 32], _: [i32; 32]) { }
// CHECK: define{{.*}}foo64{{.*}}!type ![[TYPE64:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo65(_: [i32; 32], _: [i32; 32], _: [i32; 32]) { }
// CHECK: define{{.*}}foo65{{.*}}!type ![[TYPE65:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo66(_: &[i32]) { }
// CHECK: define{{.*}}foo66{{.*}}!type ![[TYPE66:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo67(_: &[i32], _: &[i32]) { }
// CHECK: define{{.*}}foo67{{.*}}!type ![[TYPE67:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo68(_: &[i32], _: &[i32], _: &[i32]) { }
// CHECK: define{{.*}}foo68{{.*}}!type ![[TYPE68:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo69(_: &Struct1::<i32>) { }
// CHECK: define{{.*}}foo69{{.*}}!type ![[TYPE69:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo70(_: &Struct1::<i32>, _: &Struct1::<i32>) { }
// CHECK: define{{.*}}foo70{{.*}}!type ![[TYPE70:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo71(_: &Struct1::<i32>, _: &Struct1::<i32>, _: &Struct1::<i32>) { }
// CHECK: define{{.*}}foo71{{.*}}!type ![[TYPE71:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo72(_: &Enum1::<i32>) { }
// CHECK: define{{.*}}foo72{{.*}}!type ![[TYPE72:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo73(_: &Enum1::<i32>, _: &Enum1::<i32>) { }
// CHECK: define{{.*}}foo73{{.*}}!type ![[TYPE73:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo74(_: &Enum1::<i32>, _: &Enum1::<i32>, _: &Enum1::<i32>) { }
// CHECK: define{{.*}}foo74{{.*}}!type ![[TYPE74:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo75(_: &Union1::<i32>) { }
// CHECK: define{{.*}}foo75{{.*}}!type ![[TYPE75:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo76(_: &Union1::<i32>, _: &Union1::<i32>) { }
// CHECK: define{{.*}}foo76{{.*}}!type ![[TYPE76:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo77(_: &Union1::<i32>, _: &Union1::<i32>, _: &Union1::<i32>) { }
// CHECK: define{{.*}}foo77{{.*}}!type ![[TYPE77:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo78(_: *mut type1) { }
// CHECK: define{{.*}}foo78{{.*}}!type ![[TYPE78:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo79(_: *mut type1, _: *mut type1) { }
// CHECK: define{{.*}}foo79{{.*}}!type ![[TYPE79:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo80(_: *mut type1, _: *mut type1, _: *mut type1) { }
// CHECK: define{{.*}}foo80{{.*}}!type ![[TYPE80:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo81(_: &mut i32) { }
// CHECK: define{{.*}}foo81{{.*}}!type ![[TYPE81:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo82(_: &mut i32, _: &i32) { }
// CHECK: define{{.*}}foo82{{.*}}!type ![[TYPE82:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo83(_: &mut i32, _: &i32, _: &i32) { }
// CHECK: define{{.*}}foo83{{.*}}!type ![[TYPE83:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo84(_: &i32) { }
// CHECK: define{{.*}}foo84{{.*}}!type ![[TYPE84:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo85(_: &i32, _: &mut i32) { }
// CHECK: define{{.*}}foo85{{.*}}!type ![[TYPE85:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo86(_: &i32, _: &mut i32, _: &mut i32) { }
// CHECK: define{{.*}}foo86{{.*}}!type ![[TYPE86:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo87(_: *mut i32) { }
// CHECK: define{{.*}}foo87{{.*}}!type ![[TYPE87:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo88(_: *mut i32, _: *const i32) { }
// CHECK: define{{.*}}foo88{{.*}}!type ![[TYPE88:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo89(_: *mut i32, _: *const i32, _: *const i32) { }
// CHECK: define{{.*}}foo89{{.*}}!type ![[TYPE89:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo90(_: *const i32) { }
// CHECK: define{{.*}}foo90{{.*}}!type ![[TYPE90:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo91(_: *const i32, _: *mut i32) { }
// CHECK: define{{.*}}foo91{{.*}}!type ![[TYPE91:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo92(_: *const i32, _: *mut i32, _: *mut i32) { }
// CHECK: define{{.*}}foo92{{.*}}!type ![[TYPE92:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo93(_: fn(i32) -> i32) { }
// CHECK: define{{.*}}foo93{{.*}}!type ![[TYPE93:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo94(_: fn(i32) -> i32, _: fn(i32) -> i32) { }
// CHECK: define{{.*}}foo94{{.*}}!type ![[TYPE94:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo95(_: fn(i32) -> i32, _: fn(i32) -> i32, _: fn(i32) -> i32) { }
// CHECK: define{{.*}}foo95{{.*}}!type ![[TYPE95:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo96(_: &dyn Fn(i32) -> i32) { }
// CHECK: define{{.*}}foo96{{.*}}!type ![[TYPE96:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo97(_: &dyn Fn(i32) -> i32, _: &dyn Fn(i32) -> i32) { }
// CHECK: define{{.*}}foo97{{.*}}!type ![[TYPE97:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo98(_: &dyn Fn(i32) -> i32, _: &dyn Fn(i32) -> i32, _: &dyn Fn(i32) -> i32) { }
// CHECK: define{{.*}}foo98{{.*}}!type ![[TYPE98:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo99(_: &dyn FnMut(i32) -> i32) { }
// CHECK: define{{.*}}foo99{{.*}}!type ![[TYPE99:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo100(_: &dyn FnMut(i32) -> i32, _: &dyn FnMut(i32) -> i32) { }
// CHECK: define{{.*}}foo100{{.*}}!type ![[TYPE100:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo101(_: &dyn FnMut(i32) -> i32, _: &dyn FnMut(i32) -> i32, _: &dyn FnMut(i32) -> i32) { }
// CHECK: define{{.*}}foo101{{.*}}!type ![[TYPE101:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo102(_: &dyn FnOnce(i32) -> i32) { }
// CHECK: define{{.*}}foo102{{.*}}!type ![[TYPE102:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo103(_: &dyn FnOnce(i32) -> i32, _: &dyn FnOnce(i32) -> i32) { }
// CHECK: define{{.*}}foo103{{.*}}!type ![[TYPE103:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo104(_: &dyn FnOnce(i32) -> i32, _: &dyn FnOnce(i32) -> i32, _: &dyn FnOnce(i32) -> i32) {}
// CHECK: define{{.*}}foo104{{.*}}!type ![[TYPE104:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo105(_: &dyn Send) { }
// CHECK: define{{.*}}foo105{{.*}}!type ![[TYPE105:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo106(_: &dyn Send, _: &dyn Send) { }
// CHECK: define{{.*}}foo106{{.*}}!type ![[TYPE106:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo107(_: &dyn Send, _: &dyn Send, _: &dyn Send) { }
// CHECK: define{{.*}}foo107{{.*}}!type ![[TYPE107:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo108(_: Type1) { }
// CHECK: define{{.*}}foo108{{.*}}!type ![[TYPE108:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo109(_: Type1, _: Type1) { }
// CHECK: define{{.*}}foo109{{.*}}!type ![[TYPE109:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo110(_: Type1, _: Type1, _: Type1) { }
// CHECK: define{{.*}}foo110{{.*}}!type ![[TYPE110:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo111(_: Type2) { }
// CHECK: define{{.*}}foo111{{.*}}!type ![[TYPE111:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo112(_: Type2, _: Type2) { }
// CHECK: define{{.*}}foo112{{.*}}!type ![[TYPE112:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo113(_: Type2, _: Type2, _: Type2) { }
// CHECK: define{{.*}}foo113{{.*}}!type ![[TYPE113:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo114(_: Type3) { }
// CHECK: define{{.*}}foo114{{.*}}!type ![[TYPE114:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo115(_: Type3, _: Type3) { }
// CHECK: define{{.*}}foo115{{.*}}!type ![[TYPE115:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo116(_: Type3, _: Type3, _: Type3) { }
// CHECK: define{{.*}}foo116{{.*}}!type ![[TYPE116:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo117(_: Type4) { }
// CHECK: define{{.*}}foo117{{.*}}!type ![[TYPE117:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo118(_: Type4, _: Type4) { }
// CHECK: define{{.*}}foo118{{.*}}!type ![[TYPE118:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo119(_: Type4, _: Type4, _: Type4) { }
// CHECK: define{{.*}}foo119{{.*}}!type ![[TYPE119:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo120(_: Type5) { }
// CHECK: define{{.*}}foo120{{.*}}!type ![[TYPE120:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo121(_: Type5, _: Type5) { }
// CHECK: define{{.*}}foo121{{.*}}!type ![[TYPE121:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo122(_: Type5, _: Type5, _: Type5) { }
// CHECK: define{{.*}}foo122{{.*}}!type ![[TYPE122:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo123(_: Type6) { }
// CHECK: define{{.*}}foo123{{.*}}!type ![[TYPE123:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo124(_: Type6, _: Type6) { }
// CHECK: define{{.*}}foo124{{.*}}!type ![[TYPE124:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo125(_: Type6, _: Type6, _: Type6) { }
// CHECK: define{{.*}}foo125{{.*}}!type ![[TYPE125:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo126(_: Type7) { }
// CHECK: define{{.*}}foo126{{.*}}!type ![[TYPE126:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo127(_: Type7, _: Type7) { }
// CHECK: define{{.*}}foo127{{.*}}!type ![[TYPE127:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo128(_: Type7, _: Type7, _: Type7) { }
// CHECK: define{{.*}}foo128{{.*}}!type ![[TYPE128:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo129(_: Type8) { }
// CHECK: define{{.*}}foo129{{.*}}!type ![[TYPE129:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo130(_: Type8, _: Type8) { }
// CHECK: define{{.*}}foo130{{.*}}!type ![[TYPE130:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo131(_: Type8, _: Type8, _: Type8) { }
// CHECK: define{{.*}}foo131{{.*}}!type ![[TYPE131:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo132(_: Type9) { }
// CHECK: define{{.*}}foo132{{.*}}!type ![[TYPE132:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo133(_: Type9, _: Type9) { }
// CHECK: define{{.*}}foo133{{.*}}!type ![[TYPE133:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo134(_: Type9, _: Type9, _: Type9) { }
// CHECK: define{{.*}}foo134{{.*}}!type ![[TYPE134:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo135(_: Type10) { }
// CHECK: define{{.*}}foo135{{.*}}!type ![[TYPE135:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo136(_: Type10, _: Type10) { }
// CHECK: define{{.*}}foo136{{.*}}!type ![[TYPE136:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo137(_: Type10, _: Type10, _: Type10) { }
// CHECK: define{{.*}}foo137{{.*}}!type ![[TYPE137:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo138(_: Type11) { }
// CHECK: define{{.*}}foo138{{.*}}!type ![[TYPE138:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo139(_: Type11, _: Type11) { }
// CHECK: define{{.*}}foo139{{.*}}!type ![[TYPE139:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo140(_: Type11, _: Type11, _: Type11) { }
// CHECK: define{{.*}}foo140{{.*}}!type ![[TYPE140:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo141(_: Type12) { }
// CHECK: define{{.*}}foo141{{.*}}!type ![[TYPE141:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo142(_: Type12, _: Type12) { }
// CHECK: define{{.*}}foo142{{.*}}!type ![[TYPE142:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo143(_: Type12, _: Type12, _: Type12) { }
// CHECK: define{{.*}}foo143{{.*}}!type ![[TYPE143:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo144(_: Type13) { }
// CHECK: define{{.*}}foo144{{.*}}!type ![[TYPE144:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo145(_: Type13, _: Type13) { }
// CHECK: define{{.*}}foo145{{.*}}!type ![[TYPE145:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo146(_: Type13, _: Type13, _: Type13) { }
// CHECK: define{{.*}}foo146{{.*}}!type ![[TYPE146:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo147(_: Type14<Bar>) { }
// CHECK: define{{.*}}foo147{{.*}}!type ![[TYPE147:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo148(_: Type14<Bar>, _: Type14<Bar>) { }
// CHECK: define{{.*}}foo148{{.*}}!type ![[TYPE148:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo149(_: Type14<Bar>, _: Type14<Bar>, _: Type14<Bar>) { }
// CHECK: define{{.*}}foo149{{.*}}!type ![[TYPE149:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

// CHECK: ![[TYPE0]] = !{i64 0, !"_ZTSFvvE"}
// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvvvE"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFvvvvE"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFvPvE"}
// CHECK: ![[TYPE4]] = !{i64 0, !"_ZTSFvPvS_E"}
// CHECK: ![[TYPE5]] = !{i64 0, !"_ZTSFvPvS_S_E"}
// CHECK: ![[TYPE6]] = !{i64 0, !"_ZTSFvPKvE"}
// CHECK: ![[TYPE7]] = !{i64 0, !"_ZTSFvPKvS0_E"}
// CHECK: ![[TYPE8]] = !{i64 0, !"_ZTSFvPKvS0_S0_E"}
// CHECK: ![[TYPE9]] = !{i64 0, !"_ZTSFvbE"}
// CHECK: ![[TYPE10]] = !{i64 0, !"_ZTSFvbbE"}
// CHECK: ![[TYPE11]] = !{i64 0, !"_ZTSFvbbbE"}
// CHECK: ![[TYPE12]] = !{i64 0, !"_ZTSFvu2i8E"}
// CHECK: ![[TYPE13]] = !{i64 0, !"_ZTSFvu2i8S_E"}
// CHECK: ![[TYPE14]] = !{i64 0, !"_ZTSFvu2i8S_S_E"}
// CHECK: ![[TYPE15]] = !{i64 0, !"_ZTSFvu3i16E"}
// CHECK: ![[TYPE16]] = !{i64 0, !"_ZTSFvu3i16S_E"}
// CHECK: ![[TYPE17]] = !{i64 0, !"_ZTSFvu3i16S_S_E"}
// CHECK: ![[TYPE18]] = !{i64 0, !"_ZTSFvu3i32E"}
// CHECK: ![[TYPE19]] = !{i64 0, !"_ZTSFvu3i32S_E"}
// CHECK: ![[TYPE20]] = !{i64 0, !"_ZTSFvu3i32S_S_E"}
// CHECK: ![[TYPE21]] = !{i64 0, !"_ZTSFvu3i64E"}
// CHECK: ![[TYPE22]] = !{i64 0, !"_ZTSFvu3i64S_E"}
// CHECK: ![[TYPE23]] = !{i64 0, !"_ZTSFvu3i64S_S_E"}
// CHECK: ![[TYPE24]] = !{i64 0, !"_ZTSFvu4i128E"}
// CHECK: ![[TYPE25]] = !{i64 0, !"_ZTSFvu4i128S_E"}
// CHECK: ![[TYPE26]] = !{i64 0, !"_ZTSFvu4i128S_S_E"}
// CHECK: ![[TYPE27]] = !{i64 0, !"_ZTSFvu5isizeE"}
// CHECK: ![[TYPE28]] = !{i64 0, !"_ZTSFvu5isizeS_E"}
// CHECK: ![[TYPE29]] = !{i64 0, !"_ZTSFvu5isizeS_S_E"}
// CHECK: ![[TYPE30]] = !{i64 0, !"_ZTSFvu2u8E"}
// CHECK: ![[TYPE31]] = !{i64 0, !"_ZTSFvu2u8S_E"}
// CHECK: ![[TYPE32]] = !{i64 0, !"_ZTSFvu2u8S_S_E"}
// CHECK: ![[TYPE33]] = !{i64 0, !"_ZTSFvu3u16E"}
// CHECK: ![[TYPE34]] = !{i64 0, !"_ZTSFvu3u16S_E"}
// CHECK: ![[TYPE35]] = !{i64 0, !"_ZTSFvu3u16S_S_E"}
// CHECK: ![[TYPE36]] = !{i64 0, !"_ZTSFvu3u32E"}
// CHECK: ![[TYPE37]] = !{i64 0, !"_ZTSFvu3u32S_E"}
// CHECK: ![[TYPE38]] = !{i64 0, !"_ZTSFvu3u32S_S_E"}
// CHECK: ![[TYPE39]] = !{i64 0, !"_ZTSFvu3u64E"}
// CHECK: ![[TYPE40]] = !{i64 0, !"_ZTSFvu3u64S_E"}
// CHECK: ![[TYPE41]] = !{i64 0, !"_ZTSFvu3u64S_S_E"}
// CHECK: ![[TYPE42]] = !{i64 0, !"_ZTSFvu4u128E"}
// CHECK: ![[TYPE43]] = !{i64 0, !"_ZTSFvu4u128S_E"}
// CHECK: ![[TYPE44]] = !{i64 0, !"_ZTSFvu4u128S_S_E"}
// CHECK: ![[TYPE45]] = !{i64 0, !"_ZTSFvu5usizeE"}
// CHECK: ![[TYPE46]] = !{i64 0, !"_ZTSFvu5usizeS_E"}
// CHECK: ![[TYPE47]] = !{i64 0, !"_ZTSFvu5usizeS_S_E"}
// CHECK: ![[TYPE48]] = !{i64 0, !"_ZTSFvu3f32E"}
// CHECK: ![[TYPE49]] = !{i64 0, !"_ZTSFvu3f32S_E"}
// CHECK: ![[TYPE50]] = !{i64 0, !"_ZTSFvu3f32S_S_E"}
// CHECK: ![[TYPE51]] = !{i64 0, !"_ZTSFvu3f64E"}
// CHECK: ![[TYPE52]] = !{i64 0, !"_ZTSFvu3f64S_E"}
// CHECK: ![[TYPE53]] = !{i64 0, !"_ZTSFvu3f64S_S_E"}
// CHECK: ![[TYPE54]] = !{i64 0, !"_ZTSFvu4charE"}
// CHECK: ![[TYPE55]] = !{i64 0, !"_ZTSFvu4charS_E"}
// CHECK: ![[TYPE56]] = !{i64 0, !"_ZTSFvu4charS_S_E"}
// CHECK: ![[TYPE57]] = !{i64 0, !"_ZTSFvu3refIu3strEE"}
// CHECK: ![[TYPE58]] = !{i64 0, !"_ZTSFvu3refIu3strES0_E"}
// CHECK: ![[TYPE59]] = !{i64 0, !"_ZTSFvu3refIu3strES0_S0_E"}
// CHECK: ![[TYPE60]] = !{i64 0, !"_ZTSFvu5tupleIu3i32S_EE"}
// CHECK: ![[TYPE61]] = !{i64 0, !"_ZTSFvu5tupleIu3i32S_ES0_E"}
// CHECK: ![[TYPE62]] = !{i64 0, !"_ZTSFvu5tupleIu3i32S_ES0_S0_E"}
// CHECK: ![[TYPE63]] = !{i64 0, !"_ZTSFvA32u3i32E"}
// CHECK: ![[TYPE64]] = !{i64 0, !"_ZTSFvA32u3i32S0_E"}
// CHECK: ![[TYPE65]] = !{i64 0, !"_ZTSFvA32u3i32S0_S0_E"}
// CHECK: ![[TYPE66]] = !{i64 0, !"_ZTSFvu3refIu5sliceIu3i32EEE"}
// CHECK: ![[TYPE67]] = !{i64 0, !"_ZTSFvu3refIu5sliceIu3i32EES1_E"}
// CHECK: ![[TYPE68]] = !{i64 0, !"_ZTSFvu3refIu5sliceIu3i32EES1_S1_E"}
// CHECK: ![[TYPE69]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi7Struct1Iu3i32EEE"}
// CHECK: ![[TYPE70]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi7Struct1Iu3i32EES1_E"}
// CHECK: ![[TYPE71]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi7Struct1Iu3i32EES1_S1_E"}
// CHECK: ![[TYPE72]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi5Enum1Iu3i32EEE"}
// CHECK: ![[TYPE73]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi5Enum1Iu3i32EES1_E"}
// CHECK: ![[TYPE74]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi5Enum1Iu3i32EES1_S1_E"}
// CHECK: ![[TYPE75]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi6Union1Iu3i32EEE"}
// CHECK: ![[TYPE76]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi6Union1Iu3i32EES1_E"}
// CHECK: ![[TYPE77]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi6Union1Iu3i32EES1_S1_E"}
// CHECK: ![[TYPE78]] = !{i64 0, !"_ZTSFvP5type1E"}
// CHECK: ![[TYPE79]] = !{i64 0, !"_ZTSFvP5type1S0_E"}
// CHECK: ![[TYPE80]] = !{i64 0, !"_ZTSFvP5type1S0_S0_E"}
// CHECK: ![[TYPE81]] = !{i64 0, !"_ZTSFvU3mutu3refIu3i32EE"}
// CHECK: ![[TYPE82]] = !{i64 0, !"_ZTSFvU3mutu3refIu3i32ES0_E"}
// CHECK: ![[TYPE83]] = !{i64 0, !"_ZTSFvU3mutu3refIu3i32ES0_S0_E"}
// CHECK: ![[TYPE84]] = !{i64 0, !"_ZTSFvu3refIu3i32EE"}
// CHECK: ![[TYPE85]] = !{i64 0, !"_ZTSFvu3refIu3i32EU3mutS0_E"}
// CHECK: ![[TYPE86]] = !{i64 0, !"_ZTSFvu3refIu3i32EU3mutS0_S1_E"}
// CHECK: ![[TYPE87]] = !{i64 0, !"_ZTSFvPu3i32E"}
// CHECK: ![[TYPE88]] = !{i64 0, !"_ZTSFvPu3i32PKS_E"}
// CHECK: ![[TYPE89]] = !{i64 0, !"_ZTSFvPu3i32PKS_S2_E"}
// CHECK: ![[TYPE90]] = !{i64 0, !"_ZTSFvPKu3i32E"}
// CHECK: ![[TYPE91]] = !{i64 0, !"_ZTSFvPKu3i32PS_E"}
// CHECK: ![[TYPE92]] = !{i64 0, !"_ZTSFvPKu3i32PS_S2_E"}
// CHECK: ![[TYPE93]] = !{i64 0, !"_ZTSFvPFu3i32S_EE"}
// CHECK: ![[TYPE94]] = !{i64 0, !"_ZTSFvPFu3i32S_ES0_E"}
// CHECK: ![[TYPE95]] = !{i64 0, !"_ZTSFvPFu3i32S_ES0_S0_E"}
// CHECK: ![[TYPE96]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function2FnIu5paramEu6regionEEE"}
// CHECK: ![[TYPE97]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function2FnIu5paramEu6regionEES3_E"}
// CHECK: ![[TYPE98]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function2FnIu5paramEu6regionEES3_S3_E"}
// CHECK: ![[TYPE99]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function5FnMutIu5paramEu6regionEEE"}
// CHECK: ![[TYPE100]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function5FnMutIu5paramEu6regionEES3_E"}
// CHECK: ![[TYPE101]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function5FnMutIu5paramEu6regionEES3_S3_E"}
// CHECK: ![[TYPE102]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function6FnOnceIu5paramEu6regionEEE"}
// CHECK: ![[TYPE103]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function6FnOnceIu5paramEu6regionEES3_E"}
// CHECK: ![[TYPE104]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function6FnOnceIu5paramEu6regionEES3_S3_E"}
// CHECK: ![[TYPE105]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker4Sendu6regionEEE"}
// CHECK: ![[TYPE106]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker4Sendu6regionEES2_E"}
// CHECK: ![[TYPE107]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtC{{[[:print:]]+}}_4core6marker4Sendu6regionEES2_S2_E"}
// CHECK: ![[TYPE108]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NCNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn111{{[{}][{}]}}closure{{[}][}]}}Iu2i8PFvvEvEE"}
// CHECK: ![[TYPE109]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NCNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn111{{[{}][{}]}}closure{{[}][}]}}Iu2i8PFvvEvES1_E"}
// CHECK: ![[TYPE110]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NCNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn111{{[{}][{}]}}closure{{[}][}]}}Iu2i8PFvvEvES1_S1_E"}
// CHECK: ![[TYPE111]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NcNtNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn13Foo15{{[{}][{}]}}constructor{{[}][}]}}E"}
// CHECK: ![[TYPE112]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NcNtNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn13Foo15{{[{}][{}]}}constructor{{[}][}]}}S_E"}
// CHECK: ![[TYPE113]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NcNtNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn13Foo15{{[{}][{}]}}constructor{{[}][}]}}S_S_E"}
// CHECK: ![[TYPE114]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NvNFNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn110{{[{}][{}]}}extern{{[}][}]}}3fooE"}
// CHECK: ![[TYPE115]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NvNFNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn110{{[{}][{}]}}extern{{[}][}]}}3fooS_E"}
// CHECK: ![[TYPE116]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NvNFNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn110{{[{}][{}]}}extern{{[}][}]}}3fooS_S_E"}
// CHECK: ![[TYPE117]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtNCNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn1s0_11{{[{}][{}]}}closure{{[}][}]}}3FooE"}
// CHECK: ![[TYPE118]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtNCNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn1s0_11{{[{}][{}]}}closure{{[}][}]}}3FooS_E"}
// CHECK: ![[TYPE119]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtNCNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn1s0_11{{[{}][{}]}}closure{{[}][}]}}3FooS_S_E"}
// CHECK: ![[TYPE120]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtNkNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn112{{[{}][{}]}}constant{{[}][}]}}3FooE"}
// CHECK: ![[TYPE121]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtNkNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn112{{[{}][{}]}}constant{{[}][}]}}3FooS_E"}
// CHECK: ![[TYPE122]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtNkNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn112{{[{}][{}]}}constant{{[}][}]}}3FooS_S_E"}
// CHECK: ![[TYPE123]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NvNINvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn18{{[{}][{}]}}impl{{[}][}]}}3fooIu3i32EE"}
// CHECK: ![[TYPE124]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NvNINvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn18{{[{}][{}]}}impl{{[}][}]}}3fooIu3i32ES0_E"}
// CHECK: ![[TYPE125]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NvNINvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn18{{[{}][{}]}}impl{{[}][}]}}3fooIu3i32ES0_S0_E"}
// CHECK: ![[TYPE126]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NvNtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi6Trait13fooIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi6Trait1Iu5paramEu6regionEu3i32EE"}
// CHECK: ![[TYPE127]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NvNtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi6Trait13fooIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi6Trait1Iu5paramEu6regionEu3i32ES4_E"}
// CHECK: ![[TYPE128]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NvNtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi6Trait13fooIu3dynIu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi6Trait1Iu5paramEu6regionEu3i32ES4_S4_E"}
// CHECK: ![[TYPE129]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NvNtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi6Trait13fooIu3i32S_EE"}
// CHECK: ![[TYPE130]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NvNtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi6Trait13fooIu3i32S_ES0_E"}
// CHECK: ![[TYPE131]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NvNtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi6Trait13fooIu3i32S_ES0_S0_E"}
// CHECK: ![[TYPE132]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NvNtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi6Trait13fooIu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi7Struct1Iu3i32ES_EE"}
// CHECK: ![[TYPE133]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NvNtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi6Trait13fooIu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi7Struct1Iu3i32ES_ES1_E"}
// CHECK: ![[TYPE134]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NvNtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi6Trait13fooIu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi7Struct1Iu3i32ES_ES1_S1_E"}
// CHECK: ![[TYPE135]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn13QuxIu3i32Lu5usize32EEE"}
// CHECK: ![[TYPE136]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn13QuxIu3i32Lu5usize32EES2_E"}
// CHECK: ![[TYPE137]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn13QuxIu3i32Lu5usize32EES2_S2_E"}
// CHECK: ![[TYPE138]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NcNtNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn15Quuux15{{[{}][{}]}}constructor{{[}][}]}}Iu6regionS_EE"}
// CHECK: ![[TYPE139]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NcNtNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn15Quuux15{{[{}][{}]}}constructor{{[}][}]}}Iu6regionS_ES0_E"}
// CHECK: ![[TYPE140]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NcNtNvC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3fn15Quuux15{{[{}][{}]}}constructor{{[}][}]}}Iu6regionS_ES0_S0_E"}
// CHECK: ![[TYPE141]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3FooE"}
// CHECK: ![[TYPE142]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3FooS_E"}
// CHECK: ![[TYPE143]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3FooS_S_E"}
// CHECK: ![[TYPE144]] = !{i64 0, !"_ZTSFvu3refIvEE"}
// CHECK: ![[TYPE145]] = !{i64 0, !"_ZTSFvu3refIvES_E"}
// CHECK: ![[TYPE146]] = !{i64 0, !"_ZTSFvu3refIvES_S_E"}
// CHECK: ![[TYPE147]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3BarE"}
// CHECK: ![[TYPE148]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3BarS_E"}
// CHECK: ![[TYPE149]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtC{{[[:print:]]+}}_51sanitizer_cfi_emit_type_metadata_id_itanium_cxx_abi3BarS_S_E"}
