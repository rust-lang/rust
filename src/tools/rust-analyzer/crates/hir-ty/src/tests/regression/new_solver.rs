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
            174..181 '{     }': <&'a Grid as IntoIterator>::IntoIter
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
