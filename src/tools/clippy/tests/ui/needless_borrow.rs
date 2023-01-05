// run-rustfix
#![feature(custom_inner_attributes, lint_reasons, rustc_private)]
#![allow(
    unused,
    clippy::uninlined_format_args,
    clippy::unnecessary_mut_passed,
    clippy::unnecessary_to_owned
)]
#![warn(clippy::needless_borrow)]

fn main() {
    let a = 5;
    let ref_a = &a;
    let _ = x(&a); // no warning
    let _ = x(&&a); // warn

    let mut b = 5;
    mut_ref(&mut b); // no warning
    mut_ref(&mut &mut b); // warn

    let s = &String::from("hi");
    let s_ident = f(&s); // should not error, because `&String` implements Copy, but `String` does not
    let g_val = g(&Vec::new()); // should not error, because `&Vec<T>` derefs to `&[T]`
    let vec = Vec::new();
    let vec_val = g(&vec); // should not error, because `&Vec<T>` derefs to `&[T]`
    h(&"foo"); // should not error, because the `&&str` is required, due to `&Trait`
    let garbl = match 42 {
        44 => &a,
        45 => {
            println!("foo");
            &&a
        },
        46 => &&a,
        47 => {
            println!("foo");
            loop {
                println!("{}", a);
                if a == 25 {
                    break &ref_a;
                }
            }
        },
        _ => panic!(),
    };

    let _ = x(&&&a);
    let _ = x(&mut &&a);
    let _ = x(&&&mut b);
    let _ = x(&&ref_a);
    {
        let b = &mut b;
        x(&b);
    }

    // Issue #8191
    let mut x = 5;
    let mut x = &mut x;

    mut_ref(&mut x);
    mut_ref(&mut &mut x);
    let y: &mut i32 = &mut x;
    let y: &mut i32 = &mut &mut x;

    let y = match 0 {
        // Don't lint. Removing the borrow would move 'x'
        0 => &mut x,
        _ => &mut *x,
    };
    let y: &mut i32 = match 0 {
        // Lint here. The type given above triggers auto-borrow.
        0 => &mut x,
        _ => &mut *x,
    };
    fn ref_mut_i32(_: &mut i32) {}
    ref_mut_i32(match 0 {
        // Lint here. The type given above triggers auto-borrow.
        0 => &mut x,
        _ => &mut *x,
    });
    // use 'x' after to make sure it's still usable in the fixed code.
    *x = 5;

    let s = String::new();
    // let _ = (&s).len();
    // let _ = (&s).capacity();
    // let _ = (&&s).capacity();

    let x = (1, 2);
    let _ = (&x).0;
    let x = &x as *const (i32, i32);
    let _ = unsafe { (&*x).0 };

    // Issue #8367
    trait Foo {
        fn foo(self);
    }
    impl Foo for &'_ () {
        fn foo(self) {}
    }
    (&()).foo(); // Don't lint. `()` doesn't implement `Foo`
    (&&()).foo();

    impl Foo for i32 {
        fn foo(self) {}
    }
    impl Foo for &'_ i32 {
        fn foo(self) {}
    }
    (&5).foo(); // Don't lint. `5` will call `<i32 as Foo>::foo`
    (&&5).foo();

    trait FooRef {
        fn foo_ref(&self);
    }
    impl FooRef for () {
        fn foo_ref(&self) {}
    }
    impl FooRef for &'_ () {
        fn foo_ref(&self) {}
    }
    (&&()).foo_ref(); // Don't lint. `&()` will call `<() as FooRef>::foo_ref`

    struct S;
    impl From<S> for u32 {
        fn from(s: S) -> Self {
            (&s).into()
        }
    }
    impl From<&S> for u32 {
        fn from(s: &S) -> Self {
            0
        }
    }

    let _ = std::process::Command::new("ls").args(&["-a", "-l"]).status().unwrap();
    let _ = std::path::Path::new(".").join(&&".");
    deref_target_is_x(&X);
    multiple_constraints(&[[""]]);
    multiple_constraints_normalizes_to_same(&X, X);
    let _ = Some("").unwrap_or(&"");
    let _ = std::fs::write("x", &"".to_string());

    only_sized(&""); // Don't lint. `Sized` is only bound
    let _ = std::any::Any::type_id(&""); // Don't lint. `Any` is only bound
    let _ = Box::new(&""); // Don't lint. Type parameter appears in return type
    ref_as_ref_path(&""); // Don't lint. Argument type is not a type parameter
    refs_only(&()); // Don't lint. `&T` implements trait, but `T` doesn't
    multiple_constraints_normalizes_to_different(&[[""]], &[""]); // Don't lint. Projected type appears in arguments
}

#[allow(clippy::needless_borrowed_reference)]
fn x(y: &i32) -> i32 {
    *y
}

fn mut_ref(y: &mut i32) {
    *y = 5;
}

fn f<T: Copy>(y: &T) -> T {
    *y
}

fn g(y: &[u8]) -> u8 {
    y[0]
}

trait Trait {}

impl<'a> Trait for &'a str {}

fn h(_: &dyn Trait) {}

fn check_expect_suppression() {
    let a = 5;
    #[expect(clippy::needless_borrow)]
    let _ = x(&&a);
}

mod issue9160 {
    pub struct S<F> {
        f: F,
    }

    impl<T, F> S<F>
    where
        F: Fn() -> T,
    {
        fn calls_field(&self) -> T {
            (&self.f)()
        }
    }

    impl<T, F> S<F>
    where
        F: FnMut() -> T,
    {
        fn calls_mut_field(&mut self) -> T {
            (&mut self.f)()
        }
    }
}

#[derive(Clone, Copy)]
struct X;

impl std::ops::Deref for X {
    type Target = X;
    fn deref(&self) -> &Self::Target {
        self
    }
}

fn deref_target_is_x<T>(_: T)
where
    T: std::ops::Deref<Target = X>,
{
}

fn multiple_constraints<T, U, V, X, Y>(_: T)
where
    T: IntoIterator<Item = U> + IntoIterator<Item = X>,
    U: IntoIterator<Item = V>,
    V: AsRef<str>,
    X: IntoIterator<Item = Y>,
    Y: AsRef<std::ffi::OsStr>,
{
}

fn multiple_constraints_normalizes_to_same<T, U, V>(_: T, _: V)
where
    T: std::ops::Deref<Target = U>,
    U: std::ops::Deref<Target = V>,
{
}

fn only_sized<T>(_: T) {}

fn ref_as_ref_path<T: 'static>(_: &'static T)
where
    &'static T: AsRef<std::path::Path>,
{
}

trait RefsOnly {
    type Referent;
}

impl<T> RefsOnly for &T {
    type Referent = T;
}

fn refs_only<T, U>(_: T)
where
    T: RefsOnly<Referent = U>,
{
}

fn multiple_constraints_normalizes_to_different<T, U, V>(_: T, _: U)
where
    T: IntoIterator<Item = U>,
    U: IntoIterator<Item = V>,
    V: AsRef<str>,
{
}

// https://github.com/rust-lang/rust-clippy/pull/9136#pullrequestreview-1037379321
mod copyable_iterator {
    #[derive(Clone, Copy)]
    struct Iter;
    impl Iterator for Iter {
        type Item = ();
        fn next(&mut self) -> Option<Self::Item> {
            None
        }
    }
    fn takes_iter(_: impl Iterator) {}
    fn dont_warn(mut x: Iter) {
        takes_iter(&mut x);
    }
    #[allow(unused_mut)]
    fn warn(mut x: &mut Iter) {
        takes_iter(&mut x)
    }
}

#[clippy::msrv = "1.52.0"]
mod under_msrv {
    fn foo() {
        let _ = std::process::Command::new("ls").args(&["-a", "-l"]).status().unwrap();
    }
}

#[clippy::msrv = "1.53.0"]
mod meets_msrv {
    fn foo() {
        let _ = std::process::Command::new("ls").args(&["-a", "-l"]).status().unwrap();
    }
}

fn issue9383() {
    // Should not lint because unions need explicit deref when accessing field
    use std::mem::ManuallyDrop;

    union Coral {
        crab: ManuallyDrop<Vec<i32>>,
    }

    union Ocean {
        coral: ManuallyDrop<Coral>,
    }

    let mut ocean = Ocean {
        coral: ManuallyDrop::new(Coral {
            crab: ManuallyDrop::new(vec![1, 2, 3]),
        }),
    };

    unsafe {
        ManuallyDrop::drop(&mut (&mut ocean.coral).crab);

        (*ocean.coral).crab = ManuallyDrop::new(vec![4, 5, 6]);
        ManuallyDrop::drop(&mut (*ocean.coral).crab);

        ManuallyDrop::drop(&mut ocean.coral);
    }
}

fn closure_test() {
    let env = "env".to_owned();
    let arg = "arg".to_owned();
    let f = |arg| {
        let loc = "loc".to_owned();
        let _ = std::fs::write("x", &env); // Don't lint. In environment
        let _ = std::fs::write("x", &arg);
        let _ = std::fs::write("x", &loc);
    };
    let _ = std::fs::write("x", &env); // Don't lint. Borrowed by `f`
    f(arg);
}

mod significant_drop {
    #[derive(Debug)]
    struct X;

    #[derive(Debug)]
    struct Y;

    impl Drop for Y {
        fn drop(&mut self) {}
    }

    fn foo(x: X, y: Y) {
        debug(&x);
        debug(&y); // Don't lint. Has significant drop
    }

    fn debug(_: impl std::fmt::Debug) {}
}

mod used_exactly_once {
    fn foo(x: String) {
        use_x(&x);
    }
    fn use_x(_: impl AsRef<str>) {}
}

mod used_more_than_once {
    fn foo(x: String) {
        use_x(&x);
        use_x_again(&x);
    }
    fn use_x(_: impl AsRef<str>) {}
    fn use_x_again(_: impl AsRef<str>) {}
}

// https://github.com/rust-lang/rust-clippy/issues/9111#issuecomment-1277114280
mod issue_9111 {
    struct A;

    impl Extend<u8> for A {
        fn extend<T: IntoIterator<Item = u8>>(&mut self, _: T) {
            unimplemented!()
        }
    }

    impl<'a> Extend<&'a u8> for A {
        fn extend<T: IntoIterator<Item = &'a u8>>(&mut self, _: T) {
            unimplemented!()
        }
    }

    fn main() {
        let mut a = A;
        a.extend(&[]); // vs a.extend([]);
    }
}

mod issue_9710 {
    fn main() {
        let string = String::new();
        for _i in 0..10 {
            f(&string);
        }
    }

    fn f<T: AsRef<str>>(_: T) {}
}

mod issue_9739 {
    fn foo<D: std::fmt::Display>(_it: impl IntoIterator<Item = D>) {}

    fn main() {
        foo(if std::env::var_os("HI").is_some() {
            &[0]
        } else {
            &[] as &[u32]
        });
    }
}

mod issue_9739_method_variant {
    struct S;

    impl S {
        fn foo<D: std::fmt::Display>(&self, _it: impl IntoIterator<Item = D>) {}
    }

    fn main() {
        S.foo(if std::env::var_os("HI").is_some() {
            &[0]
        } else {
            &[] as &[u32]
        });
    }
}

mod issue_9782 {
    fn foo<T: AsRef<[u8]>>(t: T) {
        println!("{}", std::mem::size_of::<T>());
        let _t: &[u8] = t.as_ref();
    }

    fn main() {
        let a: [u8; 100] = [0u8; 100];

        // 100
        foo::<[u8; 100]>(a);
        foo(a);

        // 16
        foo::<&[u8]>(&a);
        foo(a.as_slice());

        // 8
        foo::<&[u8; 100]>(&a);
        foo(&a);
    }
}

mod issue_9782_type_relative_variant {
    struct S;

    impl S {
        fn foo<T: AsRef<[u8]>>(t: T) {
            println!("{}", std::mem::size_of::<T>());
            let _t: &[u8] = t.as_ref();
        }
    }

    fn main() {
        let a: [u8; 100] = [0u8; 100];

        S::foo::<&[u8; 100]>(&a);
    }
}

mod issue_9782_method_variant {
    struct S;

    impl S {
        fn foo<T: AsRef<[u8]>>(&self, t: T) {
            println!("{}", std::mem::size_of::<T>());
            let _t: &[u8] = t.as_ref();
        }
    }

    fn main() {
        let a: [u8; 100] = [0u8; 100];

        S.foo::<&[u8; 100]>(&a);
    }
}

extern crate rustc_lint;
extern crate rustc_span;

#[allow(dead_code)]
mod span_lint {
    use rustc_lint::{LateContext, Lint, LintContext};
    fn foo(cx: &LateContext<'_>, lint: &'static Lint) {
        cx.struct_span_lint(lint, rustc_span::Span::default(), "", |diag| diag.note(&String::new()));
    }
}
