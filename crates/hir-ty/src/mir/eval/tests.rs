use base_db::{fixture::WithFixture, FileId};
use hir_def::db::DefDatabase;
use syntax::{TextRange, TextSize};

use crate::{db::HirDatabase, test_db::TestDB, Interner, Substitution};

use super::{interpret_mir, MirEvalError};

fn eval_main(db: &TestDB, file_id: FileId) -> Result<(String, String), MirEvalError> {
    let module_id = db.module_for_file(file_id);
    let def_map = module_id.def_map(db);
    let scope = &def_map[module_id.local_id].scope;
    let func_id = scope
        .declarations()
        .find_map(|x| match x {
            hir_def::ModuleDefId::FunctionId(x) => {
                if db.function_data(x).name.display(db).to_string() == "main" {
                    Some(x)
                } else {
                    None
                }
            }
            _ => None,
        })
        .expect("no main function found");
    let body = db
        .monomorphized_mir_body(
            func_id.into(),
            Substitution::empty(Interner),
            db.trait_environment(func_id.into()),
        )
        .map_err(|e| MirEvalError::MirLowerError(func_id.into(), e))?;
    let (result, stdout, stderr) = interpret_mir(db, body, false, None);
    result?;
    Ok((stdout, stderr))
}

fn check_pass(ra_fixture: &str) {
    check_pass_and_stdio(ra_fixture, "", "");
}

fn check_pass_and_stdio(ra_fixture: &str, expected_stdout: &str, expected_stderr: &str) {
    let (db, file_ids) = TestDB::with_many_files(ra_fixture);
    let file_id = *file_ids.last().unwrap();
    let x = eval_main(&db, file_id);
    match x {
        Err(e) => {
            let mut err = String::new();
            let line_index = |size: TextSize| {
                let mut size = u32::from(size) as usize;
                let mut lines = ra_fixture.lines().enumerate();
                while let Some((i, l)) = lines.next() {
                    if let Some(x) = size.checked_sub(l.len()) {
                        size = x;
                    } else {
                        return (i, size);
                    }
                }
                (usize::MAX, size)
            };
            let span_formatter = |file, range: TextRange| {
                format!("{:?} {:?}..{:?}", file, line_index(range.start()), line_index(range.end()))
            };
            e.pretty_print(&mut err, &db, span_formatter).unwrap();
            panic!("Error in interpreting: {err}");
        }
        Ok((stdout, stderr)) => {
            assert_eq!(stdout, expected_stdout);
            assert_eq!(stderr, expected_stderr);
        }
    }
}

#[test]
fn function_with_extern_c_abi() {
    check_pass(
        r#"
extern "C" fn foo(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let x = foo(2, 3);
}
        "#,
    );
}

#[test]
fn drop_basic() {
    check_pass(
        r#"
//- minicore: drop, add

struct X<'a>(&'a mut i32);
impl<'a> Drop for X<'a> {
    fn drop(&mut self) {
        *self.0 += 1;
    }
}

struct NestedX<'a> { f1: X<'a>, f2: X<'a> }

fn should_not_reach() {
    _ // FIXME: replace this function with panic when that works
}

fn my_drop2(x: X<'_>) {
    return;
}

fn my_drop(x: X<'_>) {
    drop(x);
}

fn main() {
    let mut s = 10;
    let mut x = X(&mut s);
    my_drop(x);
    x = X(&mut s);
    my_drop2(x);
    X(&mut s); // dropped immediately
    let x = X(&mut s);
    NestedX { f1: x, f2: X(&mut s) };
    if s != 15 {
        should_not_reach();
    }
}
    "#,
    );
}

#[test]
fn drop_if_let() {
    check_pass(
        r#"
//- minicore: drop, add, option, cell, builtin_impls

use core::cell::Cell;

struct X<'a>(&'a Cell<i32>);
impl<'a> Drop for X<'a> {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1)
    }
}

fn should_not_reach() {
    _ // FIXME: replace this function with panic when that works
}

#[test]
fn main() {
    let s = Cell::new(0);
    let x = Some(X(&s));
    if let Some(y) = x {
        if s.get() != 0 {
            should_not_reach();
        }
        if s.get() != 0 {
            should_not_reach();
        }
    } else {
        should_not_reach();
    }
    if s.get() != 1 {
        should_not_reach();
    }
    let x = Some(X(&s));
    if let None = x {
        should_not_reach();
    } else {
        if s.get() != 1 {
            should_not_reach();
        }
    }
    if s.get() != 1 {
        should_not_reach();
    }
}
    "#,
    );
}

#[test]
fn drop_in_place() {
    check_pass(
        r#"
//- minicore: drop, add, coerce_unsized
use core::ptr::drop_in_place;

struct X<'a>(&'a mut i32);
impl<'a> Drop for X<'a> {
    fn drop(&mut self) {
        *self.0 += 1;
    }
}

fn should_not_reach() {
    _ // FIXME: replace this function with panic when that works
}

fn main() {
    let mut s = 2;
    let x = X(&mut s);
    drop_in_place(&mut x);
    drop(x);
    if s != 4 {
        should_not_reach();
    }
    let p: &mut [X] = &mut [X(&mut 2)];
    drop_in_place(p);
}
    "#,
    );
}

#[test]
fn manually_drop() {
    check_pass(
        r#"
//- minicore: manually_drop
use core::mem::ManuallyDrop;

struct X;
impl Drop for X {
    fn drop(&mut self) {
        should_not_reach();
    }
}

fn should_not_reach() {
    _ // FIXME: replace this function with panic when that works
}

fn main() {
    let x = ManuallyDrop::new(X);
}
    "#,
    );
}

#[test]
fn generic_impl_for_trait_with_generic_method() {
    check_pass(
        r#"
//- minicore: drop
struct S<T>(T);

trait Tr {
    fn f<F>(&self, x: F);
}

impl<T> Tr for S<T> {
    fn f<F>(&self, x: F) {
    }
}

fn main() {
    let s = S(1u8);
    s.f(5i64);
}
        "#,
    );
}

#[test]
fn index_of_slice_should_preserve_len() {
    check_pass(
        r#"
//- minicore: index, slice, coerce_unsized

struct X;

impl core::ops::Index<X> for [i32] {
    type Output = i32;

    fn index(&self, _: X) -> &i32 {
        if self.len() != 3 {
            should_not_reach();
        }
        &self[0]
    }
}

fn should_not_reach() {
    _ // FIXME: replace this function with panic when that works
}

fn main() {
    let x: &[i32] = &[1, 2, 3];
    &x[X];
}
        "#,
    );
}

#[test]
fn memcmp() {
    check_pass(
        r#"
//- minicore: slice, coerce_unsized, index

fn should_not_reach() -> bool {
    _ // FIXME: replace this function with panic when that works
}

extern "C" {
    fn memcmp(s1: *const u8, s2: *const u8, n: usize) -> i32;
}

fn my_cmp(x: &[u8], y: &[u8]) -> i32 {
    memcmp(x as *const u8, y as *const u8, x.len())
}

fn main() {
    if my_cmp(&[1, 2, 3], &[1, 2, 3]) != 0 {
        should_not_reach();
    }
    if my_cmp(&[1, 20, 3], &[1, 2, 3]) <= 0 {
        should_not_reach();
    }
    if my_cmp(&[1, 2, 3], &[1, 20, 3]) >= 0 {
        should_not_reach();
    }
}
    "#,
    );
}

#[test]
fn unix_write_stdout() {
    check_pass_and_stdio(
        r#"
//- minicore: slice, index, coerce_unsized

type pthread_key_t = u32;
type c_void = u8;
type c_int = i32;

extern "C" {
    pub fn write(fd: i32, buf: *const u8, count: usize) -> usize;
}

fn main() {
    let stdout = b"stdout";
    let stderr = b"stderr";
    write(1, &stdout[0], 6);
    write(2, &stderr[0], 6);
}
        "#,
        "stdout",
        "stderr",
    );
}

#[test]
fn closure_layout_in_rpit() {
    check_pass(
        r#"
//- minicore: fn

fn f<F: Fn()>(x: F) {
    fn g(x: impl Fn()) -> impl FnOnce() {
        move || {
            x();
        }
    }
    g(x)();
}

fn main() {
    f(|| {});
}
        "#,
    );
}

#[test]
fn from_fn() {
    check_pass(
        r#"
//- minicore: fn, iterator
struct FromFn<F>(F);

impl<T, F: FnMut() -> Option<T>> Iterator for FromFn<F> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        (self.0)()
    }
}

fn main() {
    let mut tokenize = {
        FromFn(move || Some(2))
    };
    let s = tokenize.next();
}
        "#,
    );
}

#[test]
fn for_loop() {
    check_pass(
        r#"
//- minicore: iterator, add
fn should_not_reach() {
    _ // FIXME: replace this function with panic when that works
}

struct X;
struct XIter(i32);

impl IntoIterator for X {
    type Item = i32;

    type IntoIter = XIter;

    fn into_iter(self) -> Self::IntoIter {
        XIter(0)
    }
}

impl Iterator for XIter {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 5 {
            None
        } else {
            self.0 += 1;
            Some(self.0)
        }
    }
}

fn main() {
    let mut s = 0;
    for x in X {
        s += x;
    }
    if s != 15 {
        should_not_reach();
    }
}
        "#,
    );
}

#[test]
fn field_with_associated_type() {
    check_pass(
        r#"
//- /b/mod.rs crate:b
pub trait Tr {
    fn f(self);
}

pub trait Tr2 {
    type Ty: Tr;
}

pub struct S<T: Tr2> {
    pub t: T::Ty,
}

impl<T: Tr2> S<T> {
    pub fn g(&self) {
        let k = (self.t, self.t);
        self.t.f();
    }
}

//- /a/mod.rs crate:a deps:b
use b::{Tr, Tr2, S};

struct A(i32);
struct B(u8);

impl Tr for A {
    fn f(&self) {
    }
}

impl Tr2 for B {
    type Ty = A;
}

#[test]
fn main() {
    let s: S<B> = S { t: A(2) };
    s.g();
}
    "#,
    );
}

#[test]
fn specialization_array_clone() {
    check_pass(
        r#"
//- minicore: copy, derive, slice, index, coerce_unsized
impl<T: Clone, const N: usize> Clone for [T; N] {
    #[inline]
    fn clone(&self) -> Self {
        SpecArrayClone::clone(self)
    }
}

trait SpecArrayClone: Clone {
    fn clone<const N: usize>(array: &[Self; N]) -> [Self; N];
}

impl<T: Clone> SpecArrayClone for T {
    #[inline]
    default fn clone<const N: usize>(array: &[T; N]) -> [T; N] {
        // FIXME: panic here when we actually implement specialization.
        from_slice(array)
    }
}

fn from_slice<T, const N: usize>(s: &[T]) -> [T; N] {
    [s[0]; N]
}

impl<T: Copy> SpecArrayClone for T {
    #[inline]
    fn clone<const N: usize>(array: &[T; N]) -> [T; N] {
        *array
    }
}

#[derive(Clone, Copy)]
struct X(i32);

fn main() {
    let ar = [X(1), X(2)];
    ar.clone();
}
        "#,
    );
}

#[test]
fn short_circuit_operator() {
    check_pass(
        r#"
fn should_not_reach() -> bool {
    _ // FIXME: replace this function with panic when that works
}

fn main() {
    if false && should_not_reach() {
        should_not_reach();
    }
    true || should_not_reach();

}
    "#,
    );
}

#[test]
fn closure_state() {
    check_pass(
        r#"
//- minicore: fn, add, copy
fn should_not_reach() {
    _ // FIXME: replace this function with panic when that works
}

fn main() {
    let mut x = 2;
    let mut c = move || {
        x += 1;
        x
    };
    c();
    c();
    c();
    if x != 2 {
        should_not_reach();
    }
    if c() != 6 {
        should_not_reach();
    }
}
        "#,
    );
}

#[test]
fn closure_capture_array_const_generic() {
    check_pass(
        r#"
//- minicore: fn, add, copy
struct X(i32);

fn f<const N: usize>(mut x: [X; N]) { // -> impl FnOnce() {
    let c = || {
        x;
    };
    c();
}

fn main() {
    let s = f([X(1)]);
    //s();
}
        "#,
    );
}

#[test]
fn syscalls() {
    check_pass(
        r#"
//- minicore: option

extern "C" {
    pub unsafe extern "C" fn syscall(num: i64, ...) -> i64;
}

const SYS_getrandom: i64 = 318;

fn should_not_reach() {
    _ // FIXME: replace this function with panic when that works
}

fn main() {
    let mut x: i32 = 0;
    let r = syscall(SYS_getrandom, &mut x, 4usize, 0);
    if r != 4 {
        should_not_reach();
    }
}

"#,
    )
}

#[test]
fn posix_tls() {
    check_pass(
        r#"
//- minicore: option

type pthread_key_t = u32;
type c_void = u8;
type c_int = i32;

extern "C" {
    pub fn pthread_key_create(
        key: *mut pthread_key_t,
        dtor: Option<unsafe extern "C" fn(*mut c_void)>,
    ) -> c_int;
    pub fn pthread_key_delete(key: pthread_key_t) -> c_int;
    pub fn pthread_getspecific(key: pthread_key_t) -> *mut c_void;
    pub fn pthread_setspecific(key: pthread_key_t, value: *const c_void) -> c_int;
}

fn main() {
    let mut key = 2;
    pthread_key_create(&mut key, None);
}
        "#,
    );
}

#[test]
fn regression_14966() {
    check_pass(
        r#"
//- minicore: fn, copy, coerce_unsized
trait A<T> {
    fn a(&self) {}
}
impl A<()> for () {}

struct B;
impl B {
    pub fn b<T>(s: &dyn A<T>) -> Self {
        B
    }
}
struct C;
impl C {
    fn c<T>(a: &dyn A<T>) -> Self {
        let mut c = C;
        let b = B::b(a);
        c.d(|| a.a());
        c
    }
    fn d(&mut self, f: impl FnOnce()) {}
}

fn main() {
    C::c(&());
}
"#,
    );
}
