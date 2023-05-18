use base_db::{fixture::WithFixture, FileId};
use hir_def::db::DefDatabase;

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
                if db.function_data(x).name.to_string() == "main" {
                    Some(x)
                } else {
                    None
                }
            }
            _ => None,
        })
        .unwrap();
    let body =
        db.mir_body(func_id.into()).map_err(|e| MirEvalError::MirLowerError(func_id.into(), e))?;
    let (result, stdout, stderr) = interpret_mir(db, &body, Substitution::empty(Interner), false);
    result?;
    Ok((stdout, stderr))
}

fn check_pass(ra_fixture: &str) {
    check_pass_and_stdio(ra_fixture, "", "");
}

fn check_pass_and_stdio(ra_fixture: &str, expected_stdout: &str, expected_stderr: &str) {
    let (db, file_id) = TestDB::with_single_file(ra_fixture);
    let x = eval_main(&db, file_id);
    match x {
        Err(e) => {
            let mut err = String::new();
            let span_formatter = |file, range| format!("{:?} {:?}", file, range);
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
