use base_db::RootQueryDb;
use chalk_ir::Substitution;
use hir_def::db::DefDatabase;
use hir_expand::EditionedFileId;
use rustc_apfloat::{
    Float,
    ieee::{Half as f16, Quad as f128},
};
use test_fixture::WithFixture;
use test_utils::skip_slow_tests;

use crate::{
    Const, ConstScalar, Interner, MemoryMap, consteval::try_const_usize, db::HirDatabase,
    display::DisplayTarget, mir::pad16, test_db::TestDB,
};

use super::{
    super::mir::{MirEvalError, MirLowerError},
    ConstEvalError,
};

mod intrinsics;

fn simplify(e: ConstEvalError) -> ConstEvalError {
    match e {
        ConstEvalError::MirEvalError(MirEvalError::InFunction(e, _)) => {
            simplify(ConstEvalError::MirEvalError(*e))
        }
        _ => e,
    }
}

#[track_caller]
fn check_fail(
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
    error: impl FnOnce(ConstEvalError) -> bool,
) {
    let (db, file_id) = TestDB::with_single_file(ra_fixture);
    match eval_goal(&db, file_id) {
        Ok(_) => panic!("Expected fail, but it succeeded"),
        Err(e) => {
            assert!(error(simplify(e.clone())), "Actual error was: {}", pretty_print_err(e, db))
        }
    }
}

#[track_caller]
fn check_number(#[rust_analyzer::rust_fixture] ra_fixture: &str, answer: i128) {
    check_answer(ra_fixture, |b, _| {
        assert_eq!(
            b,
            &answer.to_le_bytes()[0..b.len()],
            "Bytes differ. In decimal form: actual = {}, expected = {answer}",
            i128::from_le_bytes(pad16(b, true))
        );
    });
}

#[track_caller]
fn check_str(#[rust_analyzer::rust_fixture] ra_fixture: &str, answer: &str) {
    check_answer(ra_fixture, |b, mm| {
        let addr = usize::from_le_bytes(b[0..b.len() / 2].try_into().unwrap());
        let size = usize::from_le_bytes(b[b.len() / 2..].try_into().unwrap());
        let Some(bytes) = mm.get(addr, size) else {
            panic!("string data missed in the memory map");
        };
        assert_eq!(
            bytes,
            answer.as_bytes(),
            "Bytes differ. In string form: actual = {}, expected = {answer}",
            String::from_utf8_lossy(bytes)
        );
    });
}

#[track_caller]
fn check_answer(
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
    check: impl FnOnce(&[u8], &MemoryMap),
) {
    let (db, file_ids) = TestDB::with_many_files(ra_fixture);
    let file_id = *file_ids.last().unwrap();
    let r = match eval_goal(&db, file_id) {
        Ok(t) => t,
        Err(e) => {
            let err = pretty_print_err(e, db);
            panic!("Error in evaluating goal: {err}");
        }
    };
    match &r.data(Interner).value {
        chalk_ir::ConstValue::Concrete(c) => match &c.interned {
            ConstScalar::Bytes(b, mm) => {
                check(b, mm);
            }
            x => panic!("Expected number but found {x:?}"),
        },
        _ => panic!("result of const eval wasn't a concrete const"),
    }
}

fn pretty_print_err(e: ConstEvalError, db: TestDB) -> String {
    let mut err = String::new();
    let span_formatter = |file, range| format!("{file:?} {range:?}");
    let display_target =
        DisplayTarget::from_crate(&db, *db.all_crates().last().expect("no crate graph present"));
    match e {
        ConstEvalError::MirLowerError(e) => {
            e.pretty_print(&mut err, &db, span_formatter, display_target)
        }
        ConstEvalError::MirEvalError(e) => {
            e.pretty_print(&mut err, &db, span_formatter, display_target)
        }
    }
    .unwrap();
    err
}

fn eval_goal(db: &TestDB, file_id: EditionedFileId) -> Result<Const, ConstEvalError> {
    let module_id = db.module_for_file(file_id.file_id(db));
    let def_map = module_id.def_map(db);
    let scope = &def_map[module_id.local_id].scope;
    let const_id = scope
        .declarations()
        .find_map(|x| match x {
            hir_def::ModuleDefId::ConstId(x) => {
                if db.const_signature(x).name.as_ref()?.display(db, file_id.edition(db)).to_string()
                    == "GOAL"
                {
                    Some(x)
                } else {
                    None
                }
            }
            _ => None,
        })
        .expect("No const named GOAL found in the test");
    db.const_eval(const_id.into(), Substitution::empty(Interner), None)
}

#[test]
fn add() {
    check_number(r#"const GOAL: usize = 2 + 2;"#, 4);
    check_number(r#"const GOAL: i32 = -2 + --5;"#, 3);
    check_number(r#"const GOAL: i32 = 7 - 5;"#, 2);
    check_number(r#"const GOAL: i32 = 7 + (1 - 5);"#, 3);
}

#[test]
fn bit_op() {
    check_number(r#"const GOAL: u8 = !0 & !(!0 >> 1)"#, 128);
    check_number(r#"const GOAL: i8 = !0 & !(!0 >> 1)"#, 0);
    check_number(r#"const GOAL: i8 = 1 << 7"#, (1i8 << 7) as i128);
    check_number(r#"const GOAL: i8 = -1 << 2"#, (-1i8 << 2) as i128);
    check_fail(r#"const GOAL: i8 = 1 << 8"#, |e| {
        e == ConstEvalError::MirEvalError(MirEvalError::Panic("Overflow in Shl".to_owned()))
    });
    check_number(r#"const GOAL: i32 = 100000000i32 << 11"#, (100000000i32 << 11) as i128);
}

#[test]
fn floating_point() {
    check_number(
        r#"const GOAL: f128 = 2.0 + 3.0 * 5.5 - 8.;"#,
        "10.5".parse::<f128>().unwrap().to_bits() as i128,
    );
    check_number(
        r#"const GOAL: f128 = -90.0 + 36.0;"#,
        "-54.0".parse::<f128>().unwrap().to_bits() as i128,
    );
    check_number(
        r#"const GOAL: f64 = 2.0 + 3.0 * 5.5 - 8.;"#,
        i128::from_le_bytes(pad16(&f64::to_le_bytes(10.5), true)),
    );
    check_number(
        r#"const GOAL: f32 = 2.0 + 3.0 * 5.5 - 8.;"#,
        i128::from_le_bytes(pad16(&f32::to_le_bytes(10.5), true)),
    );
    check_number(
        r#"const GOAL: f32 = -90.0 + 36.0;"#,
        i128::from_le_bytes(pad16(&f32::to_le_bytes(-54.0), true)),
    );
    check_number(
        r#"const GOAL: f16 = 2.0 + 3.0 * 5.5 - 8.;"#,
        i128::from_le_bytes(pad16(
            &u16::try_from("10.5".parse::<f16>().unwrap().to_bits()).unwrap().to_le_bytes(),
            true,
        )),
    );
    check_number(
        r#"const GOAL: f16 = -90.0 + 36.0;"#,
        i128::from_le_bytes(pad16(
            &u16::try_from("-54.0".parse::<f16>().unwrap().to_bits()).unwrap().to_le_bytes(),
            true,
        )),
    );
}

#[test]
fn casts() {
    check_number(
        r#"
    //- minicore: sized
    const GOAL: usize = 12 as *const i32 as usize
        "#,
        12,
    );
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice
    const GOAL: i32 = {
        let a = [10, 20, 3, 15];
        let x: &[i32] = &a;
        let y: *const [i32] = x;
        let z = y as *const i32;
        unsafe { *z }
    };
        "#,
        10,
    );
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice
    const GOAL: i16 = {
        let a = &mut 5_i16;
        let z = a as *mut _;
        unsafe { *z }
    };
        "#,
        5,
    );
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice
    const GOAL: usize = {
        let a = &[10, 20, 30, 40] as &[i32];
        a.len()
    };
        "#,
        4,
    );
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice
    struct X {
        unsize_field: [u8],
    }

    const GOAL: usize = {
        let a = [10, 20, 3, 15];
        let x: &[i32] = &a;
        let x: *const [i32] = x;
        let x = x as *const [u8]; // slice fat pointer cast don't touch metadata
        let x = x as *const str;
        let x = x as *const X;
        let x = x as *const [i16];
        let x = x as *const X;
        let x = x as *const [u8];
        let w = unsafe { &*x };
        w.len()
    };
        "#,
        4,
    );
    check_number(
        r#"
    //- minicore: sized
    const GOAL: i32 = -12i8 as i32
        "#,
        -12,
    );
}

#[test]
fn floating_point_casts() {
    check_number(r#"const GOAL: usize = 12i32 as f32 as usize"#, 12);
    check_number(r#"const GOAL: i8 = -12i32 as f64 as i8"#, -12);
    check_number(r#"const GOAL: i32 = (-1ui8 as f32 + 2u64 as f32) as i32"#, 1);
    check_number(r#"const GOAL: i8 = (0./0.) as i8"#, 0);
    check_number(r#"const GOAL: i8 = (1./0.) as i8"#, 127);
    check_number(r#"const GOAL: i8 = (-1./0.) as i8"#, -128);
    check_number(r#"const GOAL: i64 = 1e18f64 as f32 as i64"#, 999999984306749440);
}

#[test]
fn raw_pointer_equality() {
    check_number(
        r#"
        //- minicore: copy, eq
        const GOAL: bool = {
            let a = 2;
            let p1 = a as *const i32;
            let p2 = a as *const i32;
            p1 == p2
        };
        "#,
        1,
    );
}

#[test]
fn alignment() {
    check_answer(
        r#"
//- minicore: transmute
use core::mem::transmute;
const GOAL: usize = {
    let x: i64 = 2;
    transmute(&x)
}
        "#,
        |b, _| assert_eq!(b[0] % 8, 0),
    );
    check_answer(
        r#"
//- minicore: transmute
use core::mem::transmute;
static X: i64 = 12;
const GOAL: usize = transmute(&X);
        "#,
        |b, _| assert_eq!(b[0] % 8, 0),
    );
}

#[test]
fn locals() {
    check_number(
        r#"
    const GOAL: usize = {
        let a = 3 + 2;
        let b = a * a;
        b
    };
    "#,
        25,
    );
}

#[test]
fn references() {
    check_number(
        r#"
    const GOAL: usize = {
        let x = 3;
        let y = &mut x;
        *y = 5;
        x
    };
    "#,
        5,
    );
    check_number(
        r#"
    struct Foo(i32);
    impl Foo {
        fn method(&mut self, x: i32) {
            self.0 = 2 * self.0 + x;
        }
    }
    const GOAL: i32 = {
        let mut x = Foo(3);
        x.method(5);
        x.0
    };
    "#,
        11,
    );
}

#[test]
fn reference_autoderef() {
    check_number(
        r#"
    const GOAL: usize = {
        let x = 3;
        let y = &mut x;
        let y: &mut usize = &mut y;
        *y = 5;
        x
    };
    "#,
        5,
    );
    check_number(
        r#"
    const GOAL: usize = {
        let x = 3;
        let y = &&&&&&&x;
        let z: &usize = &y;
        *z
    };
    "#,
        3,
    );
    check_number(
        r#"
    struct Foo<T> { x: T }
    impl<T> Foo<T> {
        fn foo(&mut self) -> T { self.x }
    }
    fn f(i: &mut &mut Foo<Foo<i32>>) -> i32 {
        ((**i).x).foo()
    }
    fn g(i: Foo<Foo<i32>>) -> i32 {
        i.x.foo()
    }
    const GOAL: i32 = f(&mut &mut Foo { x: Foo { x: 3 } }) + g(Foo { x: Foo { x: 5 } });
    "#,
        8,
    );
}

#[test]
fn overloaded_deref() {
    check_number(
        r#"
    //- minicore: deref_mut
    struct Foo;

    impl core::ops::Deref for Foo {
        type Target = i32;
        fn deref(&self) -> &i32 {
            &5
        }
    }

    const GOAL: i32 = {
        let x = Foo;
        let y = &*x;
        *y + *x
    };
    "#,
        10,
    );
}

#[test]
fn overloaded_deref_autoref() {
    check_number(
        r#"
    //- minicore: deref_mut
    struct Foo;
    struct Bar;

    impl core::ops::Deref for Foo {
        type Target = Bar;
        fn deref(&self) -> &Bar {
            &Bar
        }
    }

    impl Bar {
        fn method(&self) -> i32 {
            5
        }
    }

    const GOAL: i32 = Foo.method();
    "#,
        5,
    );
}

#[test]
fn overloaded_index() {
    check_number(
        r#"
    //- minicore: index
    struct Foo;

    impl core::ops::Index<usize> for Foo {
        type Output = i32;
        fn index(&self, index: usize) -> &i32 {
            if index == 7 {
                &700
            } else {
                &1000
            }
        }
    }

    impl core::ops::IndexMut<usize> for Foo {
        fn index_mut(&mut self, index: usize) -> &mut i32 {
            if index == 7 {
                &mut 7
            } else {
                &mut 10
            }
        }
    }

    const GOAL: i32 = {
        (Foo[2]) + (Foo[7]) + (*&Foo[2]) + (*&Foo[7]) + (*&mut Foo[2]) + (*&mut Foo[7])
    };
    "#,
        3417,
    );
}

#[test]
fn overloaded_binop() {
    check_number(
        r#"
    //- minicore: add
    enum Color {
        Red,
        Green,
        Yellow,
    }

    use Color::*;

    impl core::ops::Add for Color {
        type Output = Color;
        fn add(self, rhs: Color) -> Self::Output {
            Yellow
        }
    }

    impl core::ops::AddAssign for Color {
        fn add_assign(&mut self, rhs: Color) {
            *self = Red;
        }
    }

    const GOAL: bool = {
        let x = Red + Green;
        let mut y = Green;
        y += x;
        x == Yellow && y == Red && Red + Green == Yellow && Red + Red == Yellow && Yellow + Green == Yellow
    };
    "#,
        1,
    );
    check_number(
        r#"
    //- minicore: add
    impl core::ops::Add for usize {
        type Output = usize;
        fn add(self, rhs: usize) -> Self::Output {
            self + rhs
        }
    }

    impl core::ops::AddAssign for usize {
        fn add_assign(&mut self, rhs: usize) {
            *self += rhs;
        }
    }

    #[lang = "shl"]
    pub trait Shl<Rhs = Self> {
        type Output;

        fn shl(self, rhs: Rhs) -> Self::Output;
    }

    impl Shl<u8> for usize {
        type Output = usize;

        fn shl(self, rhs: u8) -> Self::Output {
            self << rhs
        }
    }

    const GOAL: usize = {
        let mut x = 10;
        x += 20;
        2 + 2 + (x << 1u8)
    };"#,
        64,
    );
}

#[test]
fn function_call() {
    check_number(
        r#"
    const fn f(x: usize) -> usize {
        2 * x + 5
    }
    const GOAL: usize = f(3);
    "#,
        11,
    );
    check_number(
        r#"
    const fn add(x: usize, y: usize) -> usize {
        x + y
    }
    const GOAL: usize = add(add(1, 2), add(3, add(4, 5)));
    "#,
        15,
    );
}

#[test]
fn trait_basic() {
    check_number(
        r#"
    trait Foo {
        fn f(&self) -> u8;
    }

    impl Foo for u8 {
        fn f(&self) -> u8 {
            *self + 33
        }
    }

    const GOAL: u8 = {
        let x = 3;
        Foo::f(&x)
    };
    "#,
        36,
    );
}

#[test]
fn trait_method() {
    check_number(
        r#"
    trait Foo {
        fn f(&self) -> u8;
    }

    impl Foo for u8 {
        fn f(&self) -> u8 {
            *self + 33
        }
    }

    const GOAL: u8 = {
        let x = 3;
        x.f()
    };
    "#,
        36,
    );
}

#[test]
fn trait_method_inside_block() {
    check_number(
        r#"
trait Twait {
    fn a(&self) -> i32;
}

fn outer() -> impl Twait {
    struct Stwuct;

    impl Twait for Stwuct {
        fn a(&self) -> i32 {
            5
        }
    }
    fn f() -> impl Twait {
        let s = Stwuct;
        s
    }
    f()
}

const GOAL: i32 = outer().a();
        "#,
        5,
    );
}

#[test]
fn generic_fn() {
    check_number(
        r#"
    trait Foo {
        fn f(&self) -> u8;
    }

    impl Foo for () {
        fn f(&self) -> u8 {
            0
        }
    }

    struct Succ<S>(S);

    impl<T: Foo> Foo for Succ<T> {
        fn f(&self) -> u8 {
            self.0.f() + 1
        }
    }

    const GOAL: u8 = Succ(Succ(())).f();
    "#,
        2,
    );
    check_number(
        r#"
    trait Foo {
        fn f(&self) -> u8;
    }

    impl Foo for u8 {
        fn f(&self) -> u8 {
            *self + 33
        }
    }

    fn foof<T: Foo>(x: T, y: T) -> u8 {
        x.f() + y.f()
    }

    const GOAL: u8 = foof(2, 5);
    "#,
        73,
    );
    check_number(
        r#"
    fn bar<A, B>(a: A, b: B) -> B {
        b
    }
        const GOAL: u8 = bar("hello", 12);
        "#,
        12,
    );
    check_number(
        r#"
        const fn y<T>(b: T) -> (T, ) {
            let alloc = b;
            (alloc, )
        }
        const GOAL: u8 = y(2).0;
        "#,
        2,
    );
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice
    fn bar<A, B>(a: A, b: B) -> B {
        b
    }
    fn foo<T>(x: [T; 2]) -> T {
        bar(x[0], x[1])
    }

    const GOAL: u8 = foo([2, 5]);
    "#,
        5,
    );
}

#[test]
fn impl_trait() {
    check_number(
        r#"
    trait Foo {
        fn f(&self) -> u8;
    }

    impl Foo for u8 {
        fn f(&self) -> u8 {
            *self + 33
        }
    }

    fn foof(x: impl Foo, y: impl Foo) -> impl Foo {
        x.f() + y.f()
    }

    const GOAL: u8 = foof(2, 5).f();
    "#,
        106,
    );
    check_number(
        r#"
        struct Foo<T>(T, T, (T, T));
        trait S {
            fn sum(&self) -> i64;
        }
        impl S for i64 {
            fn sum(&self) -> i64 {
                *self
            }
        }
        impl<T: S> S for Foo<T> {
            fn sum(&self) -> i64 {
                self.0.sum() + self.1.sum() + self.2 .0.sum() + self.2 .1.sum()
            }
        }

        fn foo() -> Foo<impl S> {
            Foo(
                Foo(1i64, 2, (3, 4)),
                Foo(5, 6, (7, 8)),
                (
                    Foo(9, 10, (11, 12)),
                    Foo(13, 14, (15, 16)),
                ),
            )
        }
        const GOAL: i64 = foo().sum();
    "#,
        136,
    );
}

#[test]
fn ifs() {
    check_number(
        r#"
    const fn f(b: bool) -> u8 {
        if b { 1 } else { 10 }
    }

    const GOAL: u8 = f(true) + f(true) + f(false);
        "#,
        12,
    );
    check_number(
        r#"
    const fn max(a: i32, b: i32) -> i32 {
        if a < b { b } else { a }
    }

    const GOAL: i32 = max(max(1, max(10, 3)), 0-122);
        "#,
        10,
    );

    check_number(
        r#"
    const fn max(a: &i32, b: &i32) -> &i32 {
        if *a < *b { b } else { a }
    }

    const GOAL: i32 = *max(max(&1, max(&10, &3)), &5);
        "#,
        10,
    );
}

#[test]
fn loops() {
    check_number(
        r#"
    const GOAL: u8 = {
        let mut x = 0;
        loop {
            x = x + 1;
            while true {
                break;
            }
            x = x + 1;
            if x == 2 {
                continue;
            }
            break;
        }
        x
    };
        "#,
        4,
    );
    check_number(
        r#"
    const GOAL: u8 = {
        let mut x = 0;
        loop {
            x = x + 1;
            if x == 5 {
                break x + 2;
            }
        }
    };
        "#,
        7,
    );
    check_number(
        r#"
    const GOAL: u8 = {
        'a: loop {
            let x = 'b: loop {
                let x = 'c: loop {
                    let x = 'd: loop {
                        let x = 'e: loop {
                            break 'd 1;
                        };
                        break 2 + x;
                    };
                    break 3 + x;
                };
                break 'a 4 + x;
            };
            break 5 + x;
        }
    };
        "#,
        8,
    );
    check_number(
        r#"
    //- minicore: add
    const GOAL: u8 = {
        let mut x = 0;
        'a: loop {
            'b: loop {
                'c: while x < 20 {
                    'd: while x < 5 {
                        'e: loop {
                            x += 1;
                            continue 'c;
                        };
                    };
                    x += 1;
                };
                break 'a;
            };
        }
        x
    };
        "#,
        20,
    );
}

#[test]
fn for_loops() {
    check_number(
        r#"
    //- minicore: iterator

    struct Range {
        start: u8,
        end: u8,
    }

    impl Iterator for Range {
        type Item = u8;
        fn next(&mut self) -> Option<u8> {
            if self.start >= self.end {
                None
            } else {
                let r = self.start;
                self.start = self.start + 1;
                Some(r)
            }
        }
    }

    const GOAL: u8 = {
        let mut sum = 0;
        let ar = Range { start: 1, end: 11 };
        for i in ar {
            sum = sum + i;
        }
        sum
    };
        "#,
        55,
    );
}

#[test]
fn ranges() {
    check_number(
        r#"
    //- minicore: range
    const GOAL: i32 = (1..2).start + (20..10).end + (100..=200).start + (2000..=1000).end
        + (10000..).start + (..100000).end + (..=1000000).end;
        "#,
        1111111,
    );
}

#[test]
fn recursion() {
    check_number(
        r#"
    const fn fact(k: i32) -> i32 {
        if k > 0 { fact(k - 1) * k } else { 1 }
    }

    const GOAL: i32 = fact(5);
        "#,
        120,
    );
}

#[test]
fn structs() {
    check_number(
        r#"
        struct Point {
            x: i32,
            y: i32,
        }

        const GOAL: i32 = {
            let p = Point { x: 5, y: 2 };
            let y = 1;
            let x = 3;
            let q = Point { y, x };
            p.x + p.y + p.x + q.y + q.y + q.x
        };
        "#,
        17,
    );
    check_number(
        r#"
        struct Point {
            x: i32,
            y: i32,
        }

        const GOAL: i32 = {
            let p = Point { x: 5, y: 2 };
            let p2 = Point { x: 3, ..p };
            p.x * 1000 + p.y * 100 + p2.x * 10 + p2.y
        };
        "#,
        5232,
    );
    check_number(
        r#"
        struct Point {
            x: i32,
            y: i32,
        }

        const GOAL: i32 = {
            let p = Point { x: 5, y: 2 };
            let Point { x, y } = p;
            let Point { x: x2, .. } = p;
            let Point { y: y2, .. } = p;
            x * 1000 + y * 100 + x2 * 10 + y2
        };
        "#,
        5252,
    );
}

#[test]
fn unions() {
    check_number(
        r#"
        union U {
            f1: i64,
            f2: (i32, i32),
        }

        const GOAL: i32 = {
            let p = U { f1: 0x0123ABCD0123DCBA };
            let p = unsafe { p.f2 };
            p.0 + p.1 + p.1
        };
        "#,
        0x0123ABCD * 2 + 0x0123DCBA,
    );
}

#[test]
fn tuples() {
    check_number(
        r#"
    const GOAL: u8 = {
        let a = (10, 20, 3, 15);
        a.1
    };
        "#,
        20,
    );
    check_number(
        r#"
    const GOAL: u8 = {
        let mut a = (10, 20, 3, 15);
        a.1 = 2;
        a.0 + a.1 + a.2 + a.3
    };
        "#,
        30,
    );
    check_number(
        r#"
    struct TupleLike(i32, i64, u8, u16);
    const GOAL: i64 = {
        let a = TupleLike(10, 20, 3, 15);
        let TupleLike(b, .., c) = a;
        a.1 * 100 + b as i64 + c as i64
    };
        "#,
        2025,
    );
    check_number(
        r#"
    const GOAL: u8 = {
        match (&(2 + 2), &4) {
            (left_val, right_val) => {
                if !(*left_val == *right_val) {
                    2
                } else {
                    5
                }
            }
        }
    };
        "#,
        5,
    );
}

#[test]
fn path_pattern_matching() {
    check_number(
        r#"
    enum Season {
        Spring,
        Summer,
        Fall,
        Winter,
    }

    use Season::*;

    const MY_SEASON: Season = Summer;

    impl Season {
        const FALL: Season = Fall;
    }

    const fn f(x: Season) -> i32 {
        match x {
            Spring => 1,
            MY_SEASON => 2,
            Season::FALL => 3,
            Winter => 4,
        }
    }
    const GOAL: i32 = f(Spring) + 10 * f(Summer) + 100 * f(Fall) + 1000 * f(Winter);
        "#,
        4321,
    );
}

#[test]
fn pattern_matching_literal() {
    check_number(
        r#"
    const fn f(x: i32) -> i32 {
        match x {
            -1 => 1,
            1 => 10,
            _ => 100,
        }
    }
    const GOAL: i32 = f(-1) + f(1) + f(0) + f(-5);
        "#,
        211,
    );
    check_number(
        r#"
    const fn f(x: &str) -> i32 {
        match x {
            "f" => 1,
            "foo" => 10,
            "" => 100,
            "bar" => 1000,
            _ => 10000,
        }
    }
    const GOAL: i32 = f("f") + f("foo") * 2 + f("") * 3 + f("bar") * 4;
        "#,
        4321,
    );
}

#[test]
fn pattern_matching_range() {
    check_number(
        r#"
    pub const L: i32 = 6;
    mod x {
        pub const R: i32 = 100;
    }
    const fn f(x: i32) -> i32 {
        match x {
            -1..=5 => x * 10,
            L..=x::R => x * 100,
            _ => x,
        }
    }
    const GOAL: i32 = f(-1) + f(2) + f(100) + f(-2) + f(1000);
        "#,
        11008,
    );
}

#[test]
fn pattern_matching_slice() {
    check_number(
        r#"
    //- minicore: slice, index, coerce_unsized, copy
    const fn f(x: &[usize]) -> usize {
        match x {
            [a, b @ .., c, d] => *a + b.len() + *c + *d,
        }
    }
    const GOAL: usize = f(&[10, 20, 3, 15, 1000, 60, 16]);
        "#,
        10 + 4 + 60 + 16,
    );
    check_number(
        r#"
    //- minicore: slice, index, coerce_unsized, copy
    const fn f(x: &[usize]) -> usize {
        match x {
            [] => 0,
            [a] => *a,
            &[a, b] => a + b,
            [a, b @ .., c, d] => *a + b.len() + *c + *d,
        }
    }
    const GOAL: usize = f(&[]) + f(&[10]) + f(&[100, 100])
        + f(&[1000, 1000, 1000]) + f(&[10000, 57, 34, 46, 10000, 10000]);
        "#,
        33213,
    );
    check_number(
        r#"
    //- minicore: slice, index, coerce_unsized, copy
    const fn f(mut slice: &[u32]) -> usize {
        slice = match slice {
            [0, rest @ ..] | rest => rest,
        };
        slice.len()
    }
    const GOAL: usize = f(&[]) + f(&[10]) + f(&[0, 100])
        + f(&[1000, 1000, 1000]) + f(&[0, 57, 34, 46, 10000, 10000]);
        "#,
        10,
    );
}

#[test]
fn pattern_matching_ergonomics() {
    check_number(
        r#"
    const fn f(x: &(u8, u8)) -> u8 {
        match x {
            (a, b) => *a + *b
        }
    }
    const GOAL: u8 = f(&(2, 3));
        "#,
        5,
    );
    check_number(
        r#"
    const GOAL: u8 = {
        let a = &(2, 3);
        let &(x, y) = a;
        x + y
    };
        "#,
        5,
    );
}

#[test]
fn destructing_assignment() {
    check_number(
        r#"
    //- minicore: add
    const fn f(i: &mut u8) -> &mut u8 {
        *i += 1;
        i
    }
    const GOAL: u8 = {
        let mut i = 4;
        _ = f(&mut i);
        i
    };
        "#,
        5,
    );
    check_number(
        r#"
    const GOAL: u8 = {
        let (mut a, mut b) = (2, 5);
        (a, b) = (b, a);
        a * 10 + b
    };
        "#,
        52,
    );
    check_number(
        r#"
    struct Point { x: i32, y: i32 }
    const GOAL: i32 = {
        let mut p = Point { x: 5, y: 6 };
        (p.x, _) = (p.y, p.x);
        p.x * 10 + p.y
    };
        "#,
        66,
    );
}

#[test]
fn let_else() {
    check_number(
        r#"
    const fn f(x: &(u8, u8)) -> u8 {
        let (a, b) = x;
        *a + *b
    }
    const GOAL: u8 = f(&(2, 3));
        "#,
        5,
    );
    check_number(
        r#"
    enum SingleVariant {
        Var(u8, u8),
    }
    const fn f(x: &&&&&SingleVariant) -> u8 {
        let SingleVariant::Var(a, b) = x;
        *a + *b
    }
    const GOAL: u8 = f(&&&&&SingleVariant::Var(2, 3));
        "#,
        5,
    );
    check_number(
        r#"
    //- minicore: option
    const fn f(x: Option<i32>) -> i32 {
        let Some(x) = x else { return 10 };
        2 * x
    }
    const GOAL: i32 = f(Some(1000)) + f(None);
        "#,
        2010,
    );
}

#[test]
fn function_param_patterns() {
    check_number(
        r#"
    const fn f((a, b): &(u8, u8)) -> u8 {
        *a + *b
    }
    const GOAL: u8 = f(&(2, 3));
        "#,
        5,
    );
    check_number(
        r#"
    const fn f(c @ (a, b): &(u8, u8)) -> u8 {
        *a + *b + c.0 + (*c).1
    }
    const GOAL: u8 = f(&(2, 3));
        "#,
        10,
    );
    check_number(
        r#"
    const fn f(ref a: u8) -> u8 {
        *a
    }
    const GOAL: u8 = f(2);
        "#,
        2,
    );
    check_number(
        r#"
    struct Foo(u8);
    impl Foo {
        const fn f(&self, (a, b): &(u8, u8)) -> u8 {
            self.0 + *a + *b
        }
    }
    const GOAL: u8 = Foo(4).f(&(2, 3));
        "#,
        9,
    );
}

#[test]
fn match_guards() {
    check_number(
        r#"
    //- minicore: option
    fn f(x: Option<i32>) -> i32 {
        match x {
            y if let Some(42) = y => 42000,
            Some(y) => y,
            None => 10
        }
    }
    const GOAL: i32 = f(Some(42)) + f(Some(2)) + f(None);
        "#,
        42012,
    );
}

#[test]
fn result_layout_niche_optimization() {
    check_number(
        r#"
    //- minicore: option, result
    const GOAL: i32 = match Some(2).ok_or(Some(2)) {
        Ok(x) => x,
        Err(_) => 1000,
    };
        "#,
        2,
    );
    check_number(
        r#"
    //- minicore: result
    pub enum AlignmentEnum64 {
        _Align1Shl0 = 1 << 0,
        _Align1Shl1 = 1 << 1,
        _Align1Shl2 = 1 << 2,
        _Align1Shl3 = 1 << 3,
        _Align1Shl4 = 1 << 4,
        _Align1Shl5 = 1 << 5,
    }
    const GOAL: Result<AlignmentEnum64, ()> = {
        let align = Err(());
        align
    };
    "#,
        0, // It is 0 since result is niche encoded and 1 is valid for `AlignmentEnum64`
    );
    check_number(
        r#"
    //- minicore: result
    pub enum AlignmentEnum64 {
        _Align1Shl0 = 1 << 0,
        _Align1Shl1 = 1 << 1,
        _Align1Shl2 = 1 << 2,
        _Align1Shl3 = 1 << 3,
        _Align1Shl4 = 1 << 4,
        _Align1Shl5 = 1 << 5,
    }
    const GOAL: i32 = {
        let align = Ok::<_, ()>(AlignmentEnum64::_Align1Shl0);
        match align {
            Ok(_) => 2,
            Err(_) => 1,
        }
    };
    "#,
        2,
    );
}

#[test]
fn options() {
    check_number(
        r#"
    //- minicore: option
    const GOAL: u8 = {
        let x = Some(2);
        match x {
            Some(y) => 2 * y,
            _ => 10,
        }
    };
        "#,
        4,
    );
    check_number(
        r#"
    //- minicore: option
    fn f(x: Option<Option<i32>>) -> i32 {
        if let Some(y) = x && let Some(z) = y {
            z
        } else if let Some(y) = x {
            1
        } else {
            0
        }
    }
    const GOAL: i32 = f(Some(Some(10))) + f(Some(None)) + f(None);
        "#,
        11,
    );
    check_number(
        r#"
    //- minicore: option
    const GOAL: u8 = {
        let x = None;
        match x {
            Some(y) => 2 * y,
            _ => 10,
        }
    };
        "#,
        10,
    );
    check_number(
        r#"
    //- minicore: option
    const GOAL: Option<&u8> = None;
        "#,
        0,
    );
}

#[test]
fn from_trait() {
    check_number(
        r#"
    //- minicore: from
    struct E1(i32);
    struct E2(i32);

    impl From<E1> for E2 {
        fn from(E1(x): E1) -> Self {
            E2(1000 * x)
        }
    }
    const GOAL: i32 = {
        let x: E2 = E1(2).into();
        x.0
    };
    "#,
        2000,
    );
}

#[test]
fn closure_clone() {
    check_number(
        r#"
//- minicore: clone, fn
struct S(u8);

impl Clone for S(u8) {
    fn clone(&self) -> S {
        S(self.0 + 5)
    }
}

const GOAL: u8 = {
    let s = S(3);
    let cl = move || s;
    let cl = cl.clone();
    cl().0
}
    "#,
        8,
    );
}

#[test]
fn builtin_derive_macro() {
    check_number(
        r#"
    //- minicore: clone, derive, builtin_impls
    #[derive(Clone)]
    enum Z {
        Foo(Y),
        Bar,
    }
    #[derive(Clone)]
    struct X(i32, Z, i64);
    #[derive(Clone)]
    struct Y {
        field1: i32,
        field2: ((i32, u8), i64),
    }

    const GOAL: u8 = {
        let x = X(2, Z::Foo(Y { field1: 4, field2: ((32, 5), 12) }), 8);
        let x = x.clone();
        let Z::Foo(t) = x.1;
        t.field2.0 .1
    };
    "#,
        5,
    );
    check_number(
        r#"
//- minicore: default, derive, builtin_impls
#[derive(Default)]
struct X(i32, Y, i64);
#[derive(Default)]
struct Y {
    field1: i32,
    field2: u8,
}

const GOAL: u8 = {
    let x = X::default();
    x.1.field2
};
"#,
        0,
    );
}

#[test]
fn try_operator() {
    check_number(
        r#"
    //- minicore: option, try
    const fn f(x: Option<i32>, y: Option<i32>) -> Option<i32> {
        Some(x? * y?)
    }
    const fn g(x: Option<i32>, y: Option<i32>) -> i32 {
        match f(x, y) {
            Some(k) => k,
            None => 5,
        }
    }
    const GOAL: i32 = g(Some(10), Some(20)) + g(Some(30), None) + g(None, Some(40)) + g(None, None);
        "#,
        215,
    );
    check_number(
        r#"
    //- minicore: result, try, from
    struct E1(i32);
    struct E2(i32);

    impl From<E1> for E2 {
        fn from(E1(x): E1) -> Self {
            E2(1000 * x)
        }
    }

    const fn f(x: Result<i32, E1>) -> Result<i32, E2> {
        Ok(x? * 10)
    }
    const fn g(x: Result<i32, E1>) -> i32 {
        match f(x) {
            Ok(k) => 7 * k,
            Err(E2(k)) => 5 * k,
        }
    }
    const GOAL: i32 = g(Ok(2)) + g(Err(E1(3)));
        "#,
        15140,
    );
}

#[test]
fn try_block() {
    check_number(
        r#"
    //- minicore: option, try
    const fn g(x: Option<i32>, y: Option<i32>) -> i32 {
        let r = try { x? * y? };
        match r {
            Some(k) => k,
            None => 5,
        }
    }
    const GOAL: i32 = g(Some(10), Some(20)) + g(Some(30), None) + g(None, Some(40)) + g(None, None);
        "#,
        215,
    );
}

#[test]
fn closures() {
    check_number(
        r#"
    //- minicore: fn, copy
    const GOAL: i32 = {
        let y = 5;
        let c = |x| x + y;
        c(2)
    };
        "#,
        7,
    );
    check_number(
        r#"
    //- minicore: fn, copy
    const GOAL: i32 = {
        let y = 5;
        let c = |(a, b): &(i32, i32)| *a + *b + y;
        c(&(2, 3))
    };
        "#,
        10,
    );
    check_number(
        r#"
    //- minicore: fn, copy
    const GOAL: i32 = {
        let mut y = 5;
        let c = |x| {
            y = y + x;
        };
        c(2);
        c(3);
        y
    };
        "#,
        10,
    );
    check_number(
        r#"
    //- minicore: fn, copy
    const GOAL: i32 = {
        let c: fn(i32) -> i32 = |x| 2 * x;
        c(2) + c(10)
    };
        "#,
        24,
    );
    check_number(
        r#"
    //- minicore: fn, copy
    struct X(i32);
    impl X {
        fn mult(&mut self, n: i32) {
            self.0 = self.0 * n
        }
    }
    const GOAL: i32 = {
        let x = X(1);
        let c = || {
            x.mult(2);
            || {
                x.mult(3);
                || {
                    || {
                        x.mult(4);
                        || {
                            x.mult(x.0);
                            || {
                                x.0
                            }
                        }
                    }
                }
            }
        };
        let r = c()()()()()();
        r + x.0
    };
        "#,
        24 * 24 * 2,
    );
}

#[test]
fn manual_fn_trait_impl() {
    check_number(
        r#"
//- minicore: fn, copy
struct S(i32);

impl FnOnce<(i32, i32)> for S {
    type Output = i32;

    extern "rust-call" fn call_once(self, arg: (i32, i32)) -> i32 {
        arg.0 + arg.1 + self.0
    }
}

const GOAL: i32 = {
    let s = S(1);
    s(2, 3)
};
"#,
        6,
    );
}

#[test]
fn closure_capture_unsized_type() {
    check_number(
        r#"
    //- minicore: fn, copy, slice, index, coerce_unsized
    fn f<T: A>(x: &<T as A>::Ty) -> &<T as A>::Ty {
        let c = || &*x;
        c()
    }

    trait A {
        type Ty;
    }

    impl A for i32 {
        type Ty = [u8];
    }

    const GOAL: u8 = {
        let k: &[u8] = &[1, 2, 3];
        let k = f::<i32>(k);
        k[0] + k[1] + k[2]
    }
    "#,
        6,
    );
}

#[test]
fn closure_and_impl_fn() {
    check_number(
        r#"
    //- minicore: fn, copy
    fn closure_wrapper<F: FnOnce() -> i32>(c: F) -> impl FnOnce() -> F {
        || c
    }

    const GOAL: i32 = {
        let y = 5;
        let c = closure_wrapper(|| y);
        c()()
    };
        "#,
        5,
    );
    check_number(
        r#"
    //- minicore: fn, copy
    fn f<T, F: Fn() -> T>(t: F) -> impl Fn() -> T {
        move || t()
    }

    const GOAL: i32 = f(|| 2)();
        "#,
        2,
    );
}

#[test]
fn or_pattern() {
    check_number(
        r#"
    const GOAL: u8 = {
        let (a | a) = 2;
        a
    };
        "#,
        2,
    );
    check_number(
        r#"
    //- minicore: option
    const fn f(x: Option<i32>) -> i32 {
        let (Some(a) | Some(a)) = x else { return 2; };
        a
    }
    const GOAL: i32 = f(Some(10)) + f(None);
        "#,
        12,
    );
    check_number(
        r#"
    //- minicore: option
    const fn f(x: Option<i32>, y: Option<i32>) -> i32 {
        match (x, y) {
            (Some(x), Some(y)) => x * y,
            (Some(a), _) | (_, Some(a)) => a,
            _ => 10,
        }
    }
    const GOAL: i32 = f(Some(10), Some(20)) + f(Some(30), None) + f(None, Some(40)) + f(None, None);
        "#,
        280,
    );
}

#[test]
fn function_pointer_in_constants() {
    check_number(
        r#"
    struct Foo {
        f: fn(u8) -> u8,
    }
    const FOO: Foo = Foo { f: add2 };
    fn add2(x: u8) -> u8 {
        x + 2
    }
    const GOAL: u8 = (FOO.f)(3);
        "#,
        5,
    );
}

#[test]
fn function_pointer_and_niche_optimization() {
    check_number(
        r#"
    //- minicore: option
    const GOAL: i32 = {
        let f: fn(i32) -> i32 = |x| x + 2;
        let init = Some(f);
        match init {
            Some(t) => t(3),
            None => 222,
        }
    };
        "#,
        5,
    );
}

#[test]
fn function_pointer() {
    check_number(
        r#"
    fn add2(x: u8) -> u8 {
        x + 2
    }
    const GOAL: u8 = {
        let plus2 = add2;
        plus2(3)
    };
        "#,
        5,
    );
    check_number(
        r#"
    fn add2(x: u8) -> u8 {
        x + 2
    }
    const GOAL: u8 = {
        let plus2: fn(u8) -> u8 = add2;
        plus2(3)
    };
        "#,
        5,
    );
    check_number(
        r#"
    //- minicore: sized
    fn add2(x: u8) -> u8 {
        x + 2
    }
    const GOAL: u8 = {
        let plus2 = add2 as fn(u8) -> u8;
        plus2(3)
    };
        "#,
        5,
    );
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice
    fn add2(x: u8) -> u8 {
        x + 2
    }
    fn mult3(x: u8) -> u8 {
        x * 3
    }
    const GOAL: u8 = {
        let x = [add2, mult3];
        x[0](1) + x[1](5)
    };
        "#,
        18,
    );
}

#[test]
fn enum_variant_as_function() {
    check_number(
        r#"
    //- minicore: option
    const GOAL: u8 = {
        let f = Some;
        f(3).unwrap_or(2)
    };
        "#,
        3,
    );
    check_number(
        r#"
    //- minicore: option
    const GOAL: u8 = {
        let f: fn(u8) -> Option<u8> = Some;
        f(3).unwrap_or(2)
    };
        "#,
        3,
    );
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice
    enum Foo {
        Add2(u8),
        Mult3(u8),
    }
    use Foo::*;
    const fn f(x: Foo) -> u8 {
        match x {
            Add2(x) => x + 2,
            Mult3(x) => x * 3,
        }
    }
    const GOAL: u8 = {
        let x = [Add2, Mult3];
        f(x[0](1)) + f(x[1](5))
    };
        "#,
        18,
    );
}

#[test]
fn function_traits() {
    check_number(
        r#"
    //- minicore: fn
    fn add2(x: u8) -> u8 {
        x + 2
    }
    fn call(f: impl Fn(u8) -> u8, x: u8) -> u8 {
        f(x)
    }
    fn call_mut(mut f: impl FnMut(u8) -> u8, x: u8) -> u8 {
        f(x)
    }
    fn call_once(f: impl FnOnce(u8) -> u8, x: u8) -> u8 {
        f(x)
    }
    const GOAL: u8 = call(add2, 3) + call_mut(add2, 3) + call_once(add2, 3);
        "#,
        15,
    );
    check_number(
        r#"
    //- minicore: coerce_unsized, fn, dispatch_from_dyn
    fn add2(x: u8) -> u8 {
        x + 2
    }
    fn call(f: &dyn Fn(u8) -> u8, x: u8) -> u8 {
        f(x)
    }
    fn call_mut(f: &mut dyn FnMut(u8) -> u8, x: u8) -> u8 {
        f(x)
    }
    const GOAL: u8 = call(&add2, 3) + call_mut(&mut add2, 3);
        "#,
        10,
    );
    check_number(
        r#"
    //- minicore: fn
    fn add2(x: u8) -> u8 {
        x + 2
    }
    fn call(f: impl Fn(u8) -> u8, x: u8) -> u8 {
        f(x)
    }
    fn call_mut(mut f: impl FnMut(u8) -> u8, x: u8) -> u8 {
        f(x)
    }
    fn call_once(f: impl FnOnce(u8) -> u8, x: u8) -> u8 {
        f(x)
    }
    const GOAL: u8 = {
        let add2: fn(u8) -> u8 = add2;
        call(add2, 3) + call_mut(add2, 3) + call_once(add2, 3)
    };
        "#,
        15,
    );
    check_number(
        r#"
    //- minicore: fn
    fn add2(x: u8) -> u8 {
        x + 2
    }
    fn call(f: &&&&&impl Fn(u8) -> u8, x: u8) -> u8 {
        f(x)
    }
    const GOAL: u8 = call(&&&&&add2, 3);
        "#,
        5,
    );
}

#[test]
fn dyn_trait() {
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice, dispatch_from_dyn
    trait Foo {
        fn foo(&self) -> u8 { 10 }
    }
    struct S1;
    struct S2;
    struct S3;
    impl Foo for S1 {
        fn foo(&self) -> u8 { 1 }
    }
    impl Foo for S2 {
        fn foo(&self) -> u8 { 2 }
    }
    impl Foo for S3 {}
    const GOAL: u8 = {
        let x: &[&dyn Foo] = &[&S1, &S2, &S3];
        x[0].foo() + x[1].foo() + x[2].foo()
    };
        "#,
        13,
    );
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice, dispatch_from_dyn
    trait Foo {
        fn foo(&self) -> i32 { 10 }
    }
    trait Bar {
        fn bar(&self) -> i32 { 20 }
    }

    struct S;
    impl Foo for S {
        fn foo(&self) -> i32 { 200 }
    }
    impl Bar for dyn Foo {
        fn bar(&self) -> i32 { 700 }
    }
    const GOAL: i32 = {
        let x: &dyn Foo = &S;
        x.bar() + x.foo()
    };
        "#,
        900,
    );
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice, dispatch_from_dyn
    trait A {
        fn x(&self) -> i32;
    }

    trait B: A {}

    impl A for i32 {
        fn x(&self) -> i32 {
            5
        }
    }

    impl B for i32 {

    }

    const fn f(x: &dyn B) -> i32 {
        x.x()
    }

    const GOAL: i32 = f(&2i32);
        "#,
        5,
    );
}

#[test]
fn coerce_unsized() {
    check_number(
        r#"
//- minicore: coerce_unsized, deref_mut, slice, index, transmute, non_null
use core::ops::{Deref, DerefMut, CoerceUnsized};
use core::{marker::Unsize, mem::transmute, ptr::NonNull};

struct ArcInner<T: ?Sized> {
    strong: usize,
    weak: usize,
    data: T,
}

pub struct Arc<T: ?Sized> {
    inner: NonNull<ArcInner<T>>,
}

impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Arc<U>> for Arc<T> {}

const GOAL: usize = {
    let x = transmute::<usize, Arc<[i32; 3]>>(12);
    let y: Arc<[i32]> = x;
    let z = transmute::<Arc<[i32]>, (usize, usize)>(y);
    z.1
};

        "#,
        3,
    );
}

#[test]
fn boxes() {
    check_number(
        r#"
//- minicore: coerce_unsized, deref_mut, slice
use core::ops::{Deref, DerefMut};
use core::{marker::Unsize, ops::CoerceUnsized};

#[lang = "owned_box"]
pub struct Box<T: ?Sized> {
    inner: *mut T,
}
impl<T> Box<T> {
    fn new(t: T) -> Self {
        #[rustc_box]
        Box::new(t)
    }
}

impl<T: ?Sized> Deref for Box<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized> DerefMut for Box<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut **self
    }
}

impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Box<U>> for Box<T> {}

const GOAL: usize = {
    let x = Box::new(5);
    let y: Box<[i32]> = Box::new([1, 2, 3]);
    *x + y.len()
};
"#,
        8,
    );
}

#[test]
fn array_and_index() {
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice
    const GOAL: u8 = {
        let a = [10, 20, 3, 15];
        let x: &[u8] = &a;
        x[1]
    };
        "#,
        20,
    );
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice
    const GOAL: usize = [1, 2, 3][2];"#,
        3,
    );
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice
    const GOAL: usize = { let a = [1, 2, 3]; let x: &[i32] = &a; x.len() };"#,
        3,
    );
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice
    const GOAL: usize = {
        let a = [1, 2, 3];
        let x: &[i32] = &a;
        let y = &*x;
        y.len()
    };"#,
        3,
    );
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice
    const GOAL: usize = [1, 2, 3, 4, 5].len();"#,
        5,
    );
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice
    const GOAL: [u16; 5] = [1, 2, 3, 4, 5];"#,
        1 + (2 << 16) + (3 << 32) + (4 << 48) + (5 << 64),
    );
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice
    const GOAL: [u16; 5] = [12; 5];"#,
        12 + (12 << 16) + (12 << 32) + (12 << 48) + (12 << 64),
    );
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice
    const LEN: usize = 4;
    const GOAL: u16 = {
        let x = [7; LEN];
        x[2]
    }"#,
        7,
    );
}

#[test]
fn string() {
    check_str(
        r#"
    //- minicore: coerce_unsized, index, slice
    const GOAL: &str = "hello";
        "#,
        "hello",
    );
}

#[test]
fn byte_string() {
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice
    const GOAL: u8 = {
        let a = b"hello";
        let x: &[u8] = a;
        x[0]
    };
        "#,
        104,
    );
}

#[test]
fn c_string() {
    check_number(
        r#"
//- minicore: index, slice
#[lang = "CStr"]
pub struct CStr {
    inner: [u8]
}
const GOAL: u8 = {
    let a = c"hello";
    a.inner[0]
};
    "#,
        104,
    );
    check_number(
        r#"
//- minicore: index, slice
#[lang = "CStr"]
pub struct CStr {
    inner: [u8]
}
const GOAL: u8 = {
    let a = c"hello";
    a.inner[6]
};
    "#,
        0,
    );
}

#[test]
fn consts() {
    check_number(
        r#"
    const F1: i32 = 1;
    const F3: i32 = 3 * F2;
    const F2: i32 = 2 * F1;
    const GOAL: i32 = F3;
    "#,
        6,
    );

    check_number(
        r#"
    const F1: i32 = 2147483647;
    const F2: i32 = F1 - 25;
    const GOAL: i32 = F2;
    "#,
        2147483622,
    );

    check_number(
        r#"
    const F1: i32 = -2147483648;
    const F2: i32 = F1 + 18;
    const GOAL: i32 = F2;
    "#,
        -2147483630,
    );

    check_number(
        r#"
    const F1: i32 = 10;
    const F2: i32 = F1 - 20;
    const GOAL: i32 = F2;
    "#,
        -10,
    );

    check_number(
        r#"
    const F1: i32 = 25;
    const F2: i32 = F1 - 25;
    const GOAL: i32 = F2;
    "#,
        0,
    );

    check_number(
        r#"
    const A: i32 = -2147483648;
    const GOAL: bool = A > 0;
    "#,
        0,
    );

    check_number(
        r#"
    const GOAL: i64 = (-2147483648_i32) as i64;
    "#,
        -2147483648,
    );
}

#[test]
fn statics() {
    check_number(
        r#"
    //- minicore: cell
    use core::cell::Cell;
    fn f() -> i32 {
        static S: Cell<i32> = Cell::new(10);
        S.set(S.get() + 1);
        S.get()
    }
    const GOAL: i32 = f() + f() + f();
    "#,
        36,
    );
}

#[test]
fn extern_weak_statics() {
    check_number(
        r#"
    //- minicore: sized
    extern "C" {
        #[linkage = "extern_weak"]
        static __dso_handle: *mut u8;
    }
    const GOAL: usize = __dso_handle as usize;
    "#,
        0,
    );
}

#[test]
// FIXME
#[should_panic]
fn from_ne_bytes() {
    check_number(
        r#"
//- minicore: int_impl
const GOAL: u32 = u32::from_ne_bytes([44, 1, 0, 0]);
        "#,
        300,
    );
}

#[test]
fn enums() {
    check_number(
        r#"
    enum E {
        F1 = 1,
        F2 = 2 * E::F1 as isize, // Rustc expects an isize here
        F3 = 3 * E::F2 as isize,
    }
    const GOAL: u8 = E::F3 as u8;
    "#,
        6,
    );
    check_number(
        r#"
    enum E { F1 = 1, F2, }
    const GOAL: u8 = E::F2 as u8;
    "#,
        2,
    );
    check_number(
        r#"
    enum E { F1, }
    const GOAL: u8 = E::F1 as u8;
    "#,
        0,
    );
    let (db, file_id) = TestDB::with_single_file(
        r#"
        enum E { A = 1, B }
        const GOAL: E = E::A;
        "#,
    );
    let r = eval_goal(&db, file_id).unwrap();
    assert_eq!(try_const_usize(&db, &r), Some(1));
}

#[test]
fn const_loop() {
    check_fail(
        r#"
    const F1: i32 = 1 * F3;
    const F3: i32 = 3 * F2;
    const F2: i32 = 2 * F1;
    const GOAL: i32 = F3;
    "#,
        |e| e == ConstEvalError::MirLowerError(MirLowerError::Loop),
    );
}

#[test]
fn const_transfer_memory() {
    check_number(
        r#"
    //- minicore: slice, index, coerce_unsized, option
    const A1: &i32 = &1;
    const A2: &i32 = &10;
    const A3: [&i32; 3] = [&1, &2, &100];
    const A4: (i32, &i32, Option<&i32>) = (1, &1000, Some(&10000));
    const GOAL: i32 = *A1 + *A2 + *A3[2] + *A4.1 + *A4.2.unwrap_or(&5);
    "#,
        11111,
    );
}

#[test]
// FIXME
#[should_panic]
fn anonymous_const_block() {
    check_number(
        r#"
    extern "rust-intrinsic" {
        pub fn size_of<T>() -> usize;
    }

    const fn f<T>() -> usize {
        let r = const { size_of::<T>() };
        r
    }

    const GOAL: usize = {
        let x = const { 2 + const { 3 } };
        let y = f::<i32>();
        x + y
    };
    "#,
        9,
    );
}

#[test]
fn const_impl_assoc() {
    check_number(
        r#"
    struct U5;
    impl U5 {
        const VAL: usize = 5;
    }
    const GOAL: usize = U5::VAL + <U5>::VAL;
    "#,
        10,
    );
}

#[test]
fn const_generic_subst_fn() {
    check_number(
        r#"
    const fn f<const A: usize>(x: usize) -> usize {
        A * x + 5
    }
    const GOAL: usize = f::<2>(3);
    "#,
        11,
    );
    check_number(
        r#"
    fn f<const N: usize>(x: [i32; N]) -> usize {
        N
    }

    trait ArrayExt {
        fn f(self) -> usize;
    }

    impl<T, const N: usize> ArrayExt for [T; N] {
        fn g(self) -> usize {
            f(self)
        }
    }

    const GOAL: usize = f([1, 2, 5]);
    "#,
        3,
    );
}

#[test]
fn layout_of_type_with_associated_type_field_defined_inside_body() {
    check_number(
        r#"
trait Tr {
    type Ty;
}

struct St<T: Tr>(T::Ty);

const GOAL: i64 = {
    // if we move `St2` out of body, the test will fail, as we don't see the impl anymore. That
    // case will probably be rejected by rustc in some later edition, but we should support this
    // case.
    struct St2;

    impl Tr for St2 {
        type Ty = i64;
    }

    struct Goal(St<St2>);

    let x = Goal(St(5));
    x.0.0
};
"#,
        5,
    );
}

#[test]
fn const_generic_subst_assoc_const_impl() {
    check_number(
        r#"
    struct Adder<const N: usize, const M: usize>;
    impl<const N: usize, const M: usize> Adder<N, M> {
        const VAL: usize = N + M;
    }
    const GOAL: usize = Adder::<2, 3>::VAL;
    "#,
        5,
    );
}

#[test]
fn associated_types() {
    check_number(
        r#"
    trait Tr {
        type Item;
        fn get_item(&self) -> Self::Item;
    }

    struct X(i32);
    struct Y(i32);

    impl Tr for X {
        type Item = Y;
        fn get_item(&self) -> Self::Item {
            Y(self.0 + 2)
        }
    }

    fn my_get_item<T: Tr>(x: T) -> <T as Tr>::Item {
        x.get_item()
    }

    const GOAL: i32 = my_get_item(X(3)).0;
    "#,
        5,
    );
}

#[test]
fn const_trait_assoc() {
    check_number(
        r#"
    struct U0;
    trait ToConst {
        const VAL: usize;
    }
    impl ToConst for U0 {
        const VAL: usize = 0;
    }
    impl ToConst for i32 {
        const VAL: usize = 32;
    }
    const GOAL: usize = U0::VAL + i32::VAL;
    "#,
        32,
    );
    check_number(
        r#"
    //- /a/lib.rs crate:a
    pub trait ToConst {
        const VAL: usize;
    }
    pub const fn to_const<T: ToConst>() -> usize {
        T::VAL
    }
    //- /main.rs crate:main deps:a
    use a::{ToConst, to_const};
    struct U0;
    impl ToConst for U0 {
        const VAL: usize = 5;
    }
    const GOAL: usize = to_const::<U0>();
    "#,
        5,
    );
    check_number(
        r#"
    //- minicore: size_of, fn
    //- /a/lib.rs crate:a
    pub struct S<T>(T);
    impl<T> S<T> {
        pub const X: usize = {
            let k: T;
            let f = || size_of::<T>();
            f()
        };
    }
    //- /main.rs crate:main deps:a
    use a::{S};
    trait Tr {
        type Ty;
    }
    impl Tr for i32 {
        type Ty = u64;
    }
    struct K<T: Tr>(<T as Tr>::Ty);
    const GOAL: usize = S::<K<i32>>::X;
    "#,
        8,
    );
    check_number(
        r#"
    //- minicore: sized
    struct S<T>(*mut T);

    trait MySized: Sized {
        const SIZE: S<Self> = S(1 as *mut Self);
    }

    impl MySized for i32 {
        const SIZE: S<i32> = S(10 as *mut i32);
    }

    impl MySized for i64 {
    }

    const fn f<T: MySized>() -> usize {
        T::SIZE.0 as usize
    }

    const GOAL: usize = f::<i32>() + f::<i64>() * 2;
    "#,
        12,
    );
}

#[test]
fn exec_limits() {
    if skip_slow_tests() {
        return;
    }

    check_fail(
        r#"
    const GOAL: usize = loop {};
    "#,
        |e| e == ConstEvalError::MirEvalError(MirEvalError::ExecutionLimitExceeded),
    );
    check_fail(
        r#"
    const fn f(x: i32) -> i32 {
        f(x + 1)
    }
    const GOAL: i32 = f(0);
    "#,
        |e| e == ConstEvalError::MirEvalError(MirEvalError::ExecutionLimitExceeded),
    );
    // Reasonable code should still work
    check_number(
        r#"
    const fn nth_odd(n: i32) -> i32 {
        2 * n - 1
    }
    const fn f(n: i32) -> i32 {
        let sum = 0;
        let i = 0;
        while i < n {
            i = i + 1;
            sum = sum + nth_odd(i);
        }
        sum
    }
    const GOAL: i32 = f(1000);
    "#,
        1000 * 1000,
    );
}

#[test]
fn memory_limit() {
    check_fail(
        r#"
        extern "Rust" {
            #[rustc_allocator]
            fn __rust_alloc(size: usize, align: usize) -> *mut u8;
        }

        const GOAL: u8 = unsafe {
            __rust_alloc(30_000_000_000, 1); // 30GB
            2
        };
        "#,
        |e| {
            e == ConstEvalError::MirEvalError(MirEvalError::Panic(
                "Memory allocation of 30000000000 bytes failed".to_owned(),
            ))
        },
    );
}

#[test]
fn type_error() {
    check_fail(
        r#"
    const GOAL: u8 = {
        let x: u16 = 2;
        let y: (u8, u8) = x;
        y.0
    };
    "#,
        |e| matches!(e, ConstEvalError::MirLowerError(MirLowerError::HasErrors)),
    );
}

#[test]
fn unsized_field() {
    check_number(
        r#"
    //- minicore: coerce_unsized, index, slice, transmute
    use core::mem::transmute;

    struct Slice([usize]);
    struct Slice2(Slice);

    impl Slice2 {
        fn as_inner(&self) -> &Slice {
            &self.0
        }

        fn as_bytes(&self) -> &[usize] {
            &self.as_inner().0
        }
    }

    const GOAL: usize = unsafe {
        let x: &[usize] = &[1, 2, 3];
        let x: &Slice2 = transmute(x);
        let x = x.as_bytes();
        x[0] + x[1] + x[2] + x.len() * 100
    };
        "#,
        306,
    );
}

#[test]
fn unsized_local() {
    check_fail(
        r#"
    //- minicore: coerce_unsized, index, slice
    const fn x() -> SomeUnknownTypeThatDereferenceToSlice {
        SomeUnknownTypeThatDereferenceToSlice
    }

    const GOAL: u16 = {
        let y = x();
        let z: &[u16] = &y;
        z[1]
    };
    "#,
        |e| matches!(e, ConstEvalError::MirLowerError(MirLowerError::UnsizedTemporary(_))),
    );
}

#[test]
fn recursive_adt() {
    check_fail(
        r#"
        //- minicore: coerce_unsized, index, slice
        pub enum TagTree {
            Leaf,
            Choice(&'static [TagTree]),
        }
        const GOAL: TagTree = {
            const TAG_TREE: TagTree = TagTree::Choice(&[
                {
                    const VARIANT_TAG_TREE: TagTree = TagTree::Choice(
                        &[
                            TAG_TREE,
                        ],
                    );
                    VARIANT_TAG_TREE
                },
            ]);
            TAG_TREE
        };
    "#,
        |e| matches!(e, ConstEvalError::MirLowerError(MirLowerError::Loop)),
    );
}
