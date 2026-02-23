use expect_test::expect;

use crate::tests::{check_infer, check_no_mismatches, check_types};

#[test]
fn regression_20365() {
    check_infer(
        r#"
//- minicore: iterator
struct Vec<T>(T);
struct IntoIter<T>(T);
impl<T> IntoIterator for Vec<T> {
    type IntoIter = IntoIter<T>;
    type Item = T;
}
impl<T> Iterator for IntoIter<T> {
    type Item = T;
}

fn f<T: Space>(a: Vec<u8>) {
    let iter = a.into_iter();
}

pub trait Space: IntoIterator {
    type Ty: Space;
}
impl Space for [u8; 1] {
    type Ty = Self;
}
    "#,
        expect![[r#"
            201..202 'a': Vec<u8>
            213..246 '{     ...r(); }': ()
            223..227 'iter': IntoIter<u8>
            230..231 'a': Vec<u8>
            230..243 'a.into_iter()': IntoIter<u8>
        "#]],
    );
}

#[test]
fn regression_19971() {
    check_infer(
        r#"
//- minicore: pointee
fn make<T>(_thin: *const (), _meta: core::ptr::DynMetadata<T>) -> *const T
where
    T: core::ptr::Pointee<Metadata = core::ptr::DynMetadata<T>> + ?Sized,
{
    loop {}
}
trait Foo {
    fn foo(&self) -> i32 {
        loop {}
    }
}

fn test() -> i32 {
    struct F {}
    impl Foo for F {}
    let meta = core::ptr::metadata(0 as *const F as *const dyn Foo);

    let f = F {};
    let fat_ptr = make(&f as *const F as *const (), meta); // <-- infers type as `*const {unknown}`

    let fat_ref = unsafe { &*fat_ptr }; // <-- infers type as `&{unknown}`
    fat_ref.foo() // cannot 'go to definition' on `foo`
}

    "#,
        expect![[r#"
            11..16 '_thin': *const ()
            29..34 '_meta': DynMetadata<T>
            155..170 '{     loop {} }': *const T
            161..168 'loop {}': !
            166..168 '{}': ()
            195..199 'self': &'? Self
            208..231 '{     ...     }': i32
            218..225 'loop {}': !
            223..225 '{}': ()
            252..613 '{     ...foo` }': i32
            300..304 'meta': DynMetadata<dyn Foo + '?>
            307..326 'core::...tadata': fn metadata<dyn Foo + '?>(*const (dyn Foo + '?)) -> <dyn Foo + '? as Pointee>::Metadata
            307..359 'core::...n Foo)': DynMetadata<dyn Foo + '?>
            327..328 '0': usize
            327..340 '0 as *const F': *const F
            327..358 '0 as *...yn Foo': *const (dyn Foo + 'static)
            370..371 'f': F
            374..378 'F {}': F
            388..395 'fat_ptr': *const (dyn Foo + '?)
            398..402 'make': fn make<dyn Foo + '?>(*const (), DynMetadata<dyn Foo + '?>) -> *const (dyn Foo + '?)
            398..437 'make(&... meta)': *const (dyn Foo + '?)
            403..405 '&f': &'? F
            403..417 '&f as *const F': *const F
            403..430 '&f as ...nst ()': *const ()
            404..405 'f': F
            432..436 'meta': DynMetadata<dyn Foo + '?>
            489..496 'fat_ref': &'? (dyn Foo + '?)
            499..519 'unsafe..._ptr }': &'? (dyn Foo + '?)
            508..517 '&*fat_ptr': &'? (dyn Foo + '?)
            509..517 '*fat_ptr': dyn Foo + '?
            510..517 'fat_ptr': *const (dyn Foo + '?)
            560..567 'fat_ref': &'? (dyn Foo + '?)
            560..573 'fat_ref.foo()': i32
        "#]],
    );
}

#[test]
fn regression_19752() {
    check_no_mismatches(
        r#"
//- minicore: sized, copy
trait T1<T: T2>: Sized + Copy {
    fn a(self, other: Self) -> Self {
        other
    }

    fn b(&mut self, other: Self) {
        *self = self.a(other);
    }
}

trait T2: Sized {
    type T1: T1<Self>;
}
    "#,
    );
}

#[test]
fn regression_type_checker_does_not_eagerly_select_predicates_from_where_clauses() {
    // This was a very long standing issue (#5514) with a lot of duplicates, that was
    // fixed by the switch to the new trait solver, so it deserves a long name and a
    // honorable mention.
    check_infer(
        r#"
//- minicore: from

struct Foo;
impl Foo {
    fn method(self) -> i32 { 0 }
}

fn f<T: Into<Foo>>(u: T) {
    let x = u.into();
    x.method();
}
    "#,
        expect![[r#"
            38..42 'self': Foo
            51..56 '{ 0 }': i32
            53..54 '0': i32
            79..80 'u': T
            85..126 '{     ...d(); }': ()
            95..96 'x': Foo
            99..100 'u': T
            99..107 'u.into()': Foo
            113..114 'x': Foo
            113..123 'x.method()': i32
        "#]],
    );
}

#[test]
fn opaque_generics() {
    check_infer(
        r#"
//- minicore: iterator
pub struct Grid {}

impl<'a> IntoIterator for &'a Grid {
    type Item = &'a ();

    type IntoIter = impl Iterator<Item = &'a ()>;

    fn into_iter(self) -> Self::IntoIter {
    }
}
    "#,
        expect![[r#"
            150..154 'self': &'a Grid
            174..181 '{     }': ()
        "#]],
    );
}

#[test]
fn normalization() {
    check_infer(
        r#"
//- minicore: iterator, iterators
fn main() {
    _ = [0i32].into_iter().filter_map(|_n| Some(1i32));
}
    "#,
        expect![[r#"
            10..69 '{     ...2)); }': ()
            16..17 '_': FilterMap<IntoIter<i32, 1>, impl FnMut(i32) -> Option<i32>>
            16..66 '_ = [0...1i32))': ()
            20..26 '[0i32]': [i32; 1]
            20..38 '[0i32]...iter()': IntoIter<i32, 1>
            20..66 '[0i32]...1i32))': FilterMap<IntoIter<i32, 1>, impl FnMut(i32) -> Option<i32>>
            21..25 '0i32': i32
            50..65 '|_n| Some(1i32)': impl FnMut(i32) -> Option<i32>
            51..53 '_n': i32
            55..59 'Some': fn Some<i32>(i32) -> Option<i32>
            55..65 'Some(1i32)': Option<i32>
            60..64 '1i32': i32
        "#]],
    );
}

#[test]
fn regression_20487() {
    check_no_mismatches(
        r#"
//- minicore: coerce_unsized, dispatch_from_dyn
trait Foo {
    fn bar(&self) -> u32 {
        0xCAFE
    }
}

fn debug(_: &dyn Foo) {}

impl Foo for i32 {}

fn main() {
    debug(&1);
}"#,
    );

    // toolchains <= 1.88.0, before sized-hierarchy.
    check_no_mismatches(
        r#"
#![feature(lang_items)]
#[lang = "sized"]
pub trait Sized {}

#[lang = "unsize"]
pub trait Unsize<T: ?Sized> {}

#[lang = "coerce_unsized"]
pub trait CoerceUnsized<T: ?Sized> {}

impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a mut U> for &'a mut T {}

impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b mut T {}

impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for &'a mut T {}

impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for &'a mut T {}

impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}

impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for &'a T {}

impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for *mut T {}

impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for *mut T {}

impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for *const T {}

#[lang = "dispatch_from_dyn"]
pub trait DispatchFromDyn<T> {}

impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<&'a U> for &'a T {}

impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<&'a mut U> for &'a mut T {}

impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<*const U> for *const T {}

impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<*mut U> for *mut T {}

trait Foo {
    fn bar(&self) -> u32 {
        0xCAFE
    }
}

fn debug(_: &dyn Foo) {}

impl Foo for i32 {}

fn main() {
    debug(&1);
}"#,
    );
}

#[test]
fn projection_is_not_associated_type() {
    check_no_mismatches(
        r#"
//- minicore: fn
trait Iterator {
    type Item;

    fn partition<F>(self, f: F)
    where
        F: FnMut(&Self::Item) -> bool,
    {
    }
}

struct Iter;
impl Iterator for Iter {
    type Item = i32;
}

fn main() {
    Iter.partition(|n| true);
}
    "#,
    );
}

#[test]
fn cast_error_type() {
    check_infer(
        r#"
fn main() {
  let foo: [_; _] = [false] as _;
}
    "#,
        expect![[r#"
            10..47 '{   le...s _; }': ()
            18..21 'foo': [bool; 1]
            32..39 '[false]': [bool; 1]
            32..44 '[false] as _': [bool; 1]
            33..38 'false': bool
        "#]],
    );
}

#[test]
fn no_infinite_loop_on_super_predicates_elaboration() {
    check_infer(
        r#"
//- minicore: sized
trait DimMax<Other: Dimension> {
    type Output: Dimension;
}

trait Dimension: DimMax<<Self as Dimension>:: Smaller, Output = Self> {
    type Smaller: Dimension;
}

fn test<T, U>(t: T)
where
    T: DimMax<U>,
    U: Dimension,
{
    let t: <T as DimMax<U>>::Output = loop {};
}
"#,
        expect![[r#"
            182..183 't': T
            230..280 '{     ... {}; }': ()
            240..241 't': <T as DimMax<U>>::Output
            270..277 'loop {}': !
            275..277 '{}': ()
        "#]],
    )
}

#[test]
fn fn_coercion() {
    check_no_mismatches(
        r#"
fn foo() {
    let _is_suffix_start: fn(&(usize, char)) -> bool = match true {
        true => |(_, c)| *c == ' ',
        _ => |(_, c)| *c == 'v',
    };
}
    "#,
    );
}

#[test]
fn coercion_with_errors() {
    check_no_mismatches(
        r#"
//- minicore: unsize, coerce_unsized
fn foo(_v: i32) -> [u8; _] { loop {} }
fn bar(_v: &[u8]) {}

fn main() {
    bar(&foo());
}
    "#,
    );
}

#[test]
fn another_20654_case() {
    check_no_mismatches(
        r#"
//- minicore: sized, unsize, coerce_unsized, dispatch_from_dyn, fn
struct Region<'db>(&'db ());

trait TypeFoldable<I: Interner> {}

trait Interner {
    type Region;
    type GenericArg;
}

struct DbInterner<'db>(&'db ());
impl<'db> Interner for DbInterner<'db> {
    type Region = Region<'db>;
    type GenericArg = GenericArg<'db>;
}

trait GenericArgExt<I: Interner<GenericArg = Self>> {
    fn expect_region(&self) -> I::Region {
        loop {}
    }
}
impl<'db> GenericArgExt<DbInterner<'db>> for GenericArg<'db> {}

enum GenericArg<'db> {
    Region(Region<'db>),
}

fn foo<'db, T: TypeFoldable<DbInterner<'db>>>(arg: GenericArg<'db>) {
    let regions = &mut || arg.expect_region();
    let f: &'_ mut (dyn FnMut() -> Region<'db> + '_) = regions;
}
    "#,
    );
}

#[test]
fn trait_solving_with_error() {
    check_infer(
        r#"
//- minicore: size_of
struct Vec<T>(T);

trait Foo {
    type Item;
    fn to_vec(self) -> Vec<Self::Item> {
        loop {}
    }
}

impl<'a, T, const N: usize> Foo for &'a [T; N] {
    type Item = T;
}

fn to_bytes() -> [u8; _] {
    loop {}
}

fn foo() {
    let _x = to_bytes().to_vec();
}
    "#,
        expect![[r#"
            60..64 'self': Self
            85..108 '{     ...     }': Vec<<Self as Foo>::Item>
            95..102 'loop {}': !
            100..102 '{}': ()
            208..223 '{     loop {} }': [u8; _]
            214..221 'loop {}': !
            219..221 '{}': ()
            234..271 '{     ...c(); }': ()
            244..246 '_x': {unknown}
            249..257 'to_bytes': fn to_bytes() -> [u8; _]
            249..259 'to_bytes()': [u8; _]
            249..268 'to_byt..._vec()': Vec<<[u8; _] as Foo>::Item>
        "#]],
    );
}

#[test]
fn regression_21315() {
    check_infer(
        r#"
struct Consts;
impl Consts { const MAX: usize = 0; }

struct Between<const M: usize, const N: usize, T>(T);

impl<const M: usize, T> Between<M, { Consts::MAX }, T> {
    fn sep_once(self, _sep: &str, _other: Self) -> Self {
        self
    }
}

trait Parser: Sized {
    fn at_least<const M: usize>(self) -> Between<M, { Consts::MAX }, Self> {
        Between(self)
    }
    fn at_most<const N: usize>(self) -> Between<0, N, Self> {
        Between(self)
    }
}

impl Parser for char {}

fn test_at_least() {
    let num = '9'.at_least::<1>();
    let _ver = num.sep_once(".", num);
}

fn test_at_most() {
    let num = '9'.at_most::<1>();
}
    "#,
        expect![[r#"
            48..49 '0': usize
            182..186 'self': Between<M, _, T>
            188..192 '_sep': &'? str
            200..206 '_other': Between<M, _, T>
            222..242 '{     ...     }': Between<M, _, T>
            232..236 'self': Between<M, _, T>
            300..304 'self': Self
            343..372 '{     ...     }': Between<M, _, Self>
            353..360 'Between': fn Between<M, _, Self>(Self) -> Between<M, _, Self>
            353..366 'Between(self)': Between<M, _, Self>
            361..365 'self': Self
            404..408 'self': Self
            433..462 '{     ...     }': Between<0, N, Self>
            443..450 'Between': fn Between<0, N, Self>(Self) -> Between<0, N, Self>
            443..456 'Between(self)': Between<0, N, Self>
            451..455 'self': Self
            510..587 '{     ...um); }': ()
            520..523 'num': Between<1, _, char>
            526..529 ''9'': char
            526..545 ''9'.at...:<1>()': Between<1, _, char>
            555..559 '_ver': Between<1, _, char>
            562..565 'num': Between<1, _, char>
            562..584 'num.se..., num)': Between<1, _, char>
            575..578 '"."': &'static str
            580..583 'num': Between<1, _, char>
            607..644 '{     ...>(); }': ()
            617..620 'num': Between<0, 1, char>
            623..626 ''9'': char
            623..641 ''9'.at...:<1>()': Between<0, 1, char>
        "#]],
    );
}

#[test]
fn regression_19637() {
    check_no_mismatches(
        r#"
//- minicore: coerce_unsized
pub trait Any {}

impl<T: 'static> Any for T {}

pub trait Trait: Any {
    type F;
}

pub struct TT {}

impl Trait for TT {
    type F = f32;
}

pub fn coercion(x: &mut dyn Any) -> &mut dyn Any {
    x
}

fn main() {
    let mut t = TT {};
    let tt = &mut t as &mut dyn Trait<F = f32>;
    let st = coercion(tt);
}
    "#,
    );
}

#[test]
fn double_into_iter() {
    check_types(
        r#"
//- minicore: iterator

fn intoiter_issue<A, B>(foo: A)
where
    A: IntoIterator<Item = B>,
    B: IntoIterator<Item = usize>,
{
    for x in foo {
    //  ^ B
        for m in x {
        //  ^ usize
        }
    }
}
"#,
    );
}

#[test]
fn regression_16282() {
    check_infer(
        r#"
//- minicore: coerce_unsized, dispatch_from_dyn
trait MapLookup<Q> {
    type MapValue;
}

impl<K> MapLookup<K> for K {
    type MapValue = K;
}

trait Map: MapLookup<<Self as Map>::Key> {
    type Key;
}

impl<K> Map for K {
    type Key = K;
}


fn main() {
    let _ = &()
        as &dyn Map<Key=u32,MapValue=u32>;
}
"#,
        expect![[r#"
            210..272 '{     ...32>; }': ()
            220..221 '_': &'? (dyn Map<MapValue = u32, Key = u32> + '?)
            224..227 '&()': &'? ()
            224..269 '&()   ...e=u32>': &'? (dyn Map<MapValue = u32, Key = u32> + 'static)
            225..227 '()': ()
        "#]],
    );
}

#[test]
fn regression_18692() {
    check_no_mismatches(
        r#"
//- minicore: coerce_unsized, dispatch_from_dyn, send
trait Trait: Send {}

fn f(_: *const (dyn Trait + Send)) {}
fn g(it: *const (dyn Trait)) {
    f(it);
}
"#,
    );
}

#[test]
fn regression_20951() {
    check_infer(
        r#"
//- minicore: async_fn
trait DoesSomething {
    fn do_something(&self) -> impl Future<Output = usize>;
}

impl<F> DoesSomething for F
where
    F: AsyncFn() -> usize,
{
    fn do_something(&self) -> impl Future<Output = usize> {
        self()
    }
}
"#,
        expect![[r#"
            43..47 'self': &'? Self
            168..172 'self': &'? F
            205..227 '{     ...     }': <F as AsyncFnMut<()>>::CallRefFuture<'<erased>>
            215..219 'self': &'? F
            215..221 'self()': <F as AsyncFnMut<()>>::CallRefFuture<'<erased>>
        "#]],
    );
}

#[test]
fn regression_19957() {
    // This test documents issue #19957: async-trait patterns incorrectly produce
    // type mismatches between Pin<Box<dyn Future>> and Pin<Box<impl Future>>.
    check_no_mismatches(
        r#"
//- minicore: future, pin, result, error, send, coerce_unsized, dispatch_from_dyn
use core::{future::Future, pin::Pin};

#[lang = "owned_box"]
pub struct Box<T: ?Sized> {
    inner: *mut T,
}

impl<T> Box<T> {
    fn pin(value: T) -> Pin<Box<T>> {
        // Implementation details don't matter here for type checking
        loop {}
    }
}

impl<T: ?Sized + core::marker::Unsize<U>, U: ?Sized> core::ops::CoerceUnsized<Box<U>> for Box<T> {}

impl<T: ?Sized + core::ops::DispatchFromDyn<U>, U: ?Sized> core::ops::DispatchFromDyn<Box<U>> for Box<T> {}

pub struct ExampleData {
    pub id: i32,
}

// Simulates what #[async_trait] expands to
pub trait SimpleModel {
    fn save<'life0, 'async_trait>(
        &'life0 self,
    ) -> Pin<Box<dyn Future<Output = i32> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        Self: 'async_trait;
}

impl SimpleModel for ExampleData {
    fn save<'life0, 'async_trait>(
        &'life0 self,
    ) -> Pin<Box<dyn Future<Output = i32> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        Self: 'async_trait,
    {
        // Body creates Pin<Box<impl Future>>, which should coerce to Pin<Box<dyn Future>>
        Box::pin(async move { self.id })
    }
}
"#,
    )
}

#[test]
fn regression_20975() {
    check_infer(
        r#"
//- minicore: future, iterators, range
use core::future::Future;

struct Foo<T>(T);

trait X {}

impl X for i32 {}
impl X for i64 {}

impl<T: X> Iterator for Foo<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.next_spec()
    }
}

trait Bar {
    type Item;

    fn next_spec(&mut self) -> Option<Self::Item>;
}

impl<T: X> Bar for Foo<T> {
    type Item = T;

    fn next_spec(&mut self) -> Option<Self::Item> {
        None
    }
}

struct JoinAll<F>
where
    F: Future,
{
    f: F,
}

fn join_all<I>(iter: I) -> JoinAll<<I as IntoIterator>::Item>
where
    I: IntoIterator,
    <I as IntoIterator>::Item: Future,
{
    loop {}
}

fn main() {
    let x = Foo(42).filter_map(|_| Some(async {}));
    join_all(x);
}
"#,
        expect![[r#"
            164..168 'self': &'? mut Foo<T>
            192..224 '{     ...     }': Option<T>
            202..206 'self': &'? mut Foo<T>
            202..218 'self.n...spec()': Option<T>
            278..282 'self': &'? mut Self
            380..384 'self': &'? mut Foo<T>
            408..428 '{     ...     }': Option<T>
            418..422 'None': Option<T>
            501..505 'iter': I
            614..629 '{     loop {} }': JoinAll<impl Future>
            620..627 'loop {}': !
            625..627 '{}': ()
            641..713 '{     ...(x); }': ()
            651..652 'x': FilterMap<Foo<i32>, impl FnMut(i32) -> Option<impl Future<Output = ()>>>
            655..658 'Foo': fn Foo<i32>(i32) -> Foo<i32>
            655..662 'Foo(42)': Foo<i32>
            655..693 'Foo(42...c {}))': FilterMap<Foo<i32>, impl FnMut(i32) -> Option<impl Future<Output = ()>>>
            659..661 '42': i32
            674..692 '|_| So...nc {})': impl FnMut(i32) -> Option<impl Future<Output = ()>>
            675..676 '_': i32
            678..682 'Some': fn Some<impl Future<Output = ()>>(impl Future<Output = ()>) -> Option<impl Future<Output = ()>>
            678..692 'Some(async {})': Option<impl Future<Output = ()>>
            683..691 'async {}': impl Future<Output = ()>
            699..707 'join_all': fn join_all<FilterMap<Foo<i32>, impl FnMut(i32) -> Option<impl Future<Output = ()>>>>(FilterMap<Foo<i32>, impl FnMut(i32) -> Option<impl Future<Output = ()>>>) -> JoinAll<<FilterMap<Foo<i32>, impl FnMut(i32) -> Option<impl Future<Output = ()>>> as IntoIterator>::Item>
            699..710 'join_all(x)': JoinAll<impl Future<Output = ()>>
            708..709 'x': FilterMap<Foo<i32>, impl FnMut(i32) -> Option<impl Future<Output = ()>>>
        "#]],
    );
}

#[test]
fn regression_19339() {
    check_infer(
        r#"
trait Bar {
    type Baz;

    fn baz(&self) -> Self::Baz;
}

trait Foo {
    type Bar;

    fn bar(&self) -> Self::Bar;
}

trait FooFactory {
    type Output: Foo<Bar: Bar<Baz = u8>>;

    fn foo(&self) -> Self::Output;

    fn foo_rpit(&self) -> impl Foo<Bar: Bar<Baz = u8>>;
}

fn test1(foo: impl Foo<Bar: Bar<Baz = u8>>) {
    let baz = foo.bar().baz();
}

fn test2<T: FooFactory>(factory: T) {
    let baz = factory.foo().bar().baz();
    let baz = factory.foo_rpit().bar().baz();
}
"#,
        expect![[r#"
            39..43 'self': &'? Self
            101..105 'self': &'? Self
            198..202 'self': &'? Self
            239..243 'self': &'? Self
            290..293 'foo': impl Foo + ?Sized
            325..359 '{     ...z(); }': ()
            335..338 'baz': u8
            341..344 'foo': impl Foo + ?Sized
            341..350 'foo.bar()': impl Bar
            341..356 'foo.bar().baz()': u8
            385..392 'factory': T
            397..487 '{     ...z(); }': ()
            407..410 'baz': u8
            413..420 'factory': T
            413..426 'factory.foo()': <T as FooFactory>::Output
            413..432 'factor....bar()': <<T as FooFactory>::Output as Foo>::Bar
            413..438 'factor....baz()': u8
            448..451 'baz': u8
            454..461 'factory': T
            454..472 'factor...rpit()': impl Foo + Bar<Baz = u8> + ?Sized
            454..478 'factor....bar()': <impl Foo + Bar<Baz = u8> + ?Sized as Foo>::Bar
            454..484 'factor....baz()': u8
        "#]],
    );
}
