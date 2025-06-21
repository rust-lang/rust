#![allow(
    unused,
    non_local_definitions,
    clippy::uninlined_format_args,
    clippy::unnecessary_mut_passed,
    clippy::unnecessary_to_owned,
    clippy::unnecessary_literal_unwrap,
    clippy::needless_lifetimes
)]
#![warn(clippy::needless_borrow)]

fn main() {
    let a = 5;
    let ref_a = &a;
    let _ = x(&a); // no warning
    let _ = x(&&a); // warn
    //
    //~^^ needless_borrow

    let mut b = 5;
    mut_ref(&mut b); // no warning
    mut_ref(&mut &mut b); // warn
    //
    //~^^ needless_borrow

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
            //~^ needless_borrow
        },
        46 => &&a,
        //~^ needless_borrow
        47 => {
            println!("foo");
            loop {
                println!("{}", a);
                if a == 25 {
                    break &ref_a;
                    //~^ needless_borrow
                }
            }
        },
        _ => panic!(),
    };

    let _ = x(&&&a);
    //~^ needless_borrow
    let _ = x(&mut &&a);
    //~^ needless_borrow
    let _ = x(&&&mut b);
    //~^ needless_borrow
    let _ = x(&&ref_a);
    //~^ needless_borrow
    {
        let b = &mut b;
        x(&b);
        //~^ needless_borrow
    }

    // Issue #8191
    let mut x = 5;
    let mut x = &mut x;

    mut_ref(&mut x);
    //~^ needless_borrow
    mut_ref(&mut &mut x);
    //~^ needless_borrow
    let y: &mut i32 = &mut x;
    //~^ needless_borrow
    let y: &mut i32 = &mut &mut x;
    //~^ needless_borrow

    let y = match 0 {
        // Don't lint. Removing the borrow would move 'x'
        0 => &mut x,
        _ => &mut *x,
    };
    let y: &mut i32 = match 0 {
        // Lint here. The type given above triggers auto-borrow.
        0 => &mut x,
        //~^ needless_borrow
        _ => &mut *x,
    };
    fn ref_mut_i32(_: &mut i32) {}
    ref_mut_i32(match 0 {
        // Lint here. The type given above triggers auto-borrow.
        0 => &mut x,
        //~^ needless_borrow
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
    //~^ needless_borrow

    // Issue #8367
    trait Foo {
        fn foo(self);
    }
    impl Foo for &'_ () {
        fn foo(self) {}
    }
    (&()).foo(); // Don't lint. `()` doesn't implement `Foo`
    (&&()).foo();
    //~^ needless_borrow

    impl Foo for i32 {
        fn foo(self) {}
    }
    impl Foo for &'_ i32 {
        fn foo(self) {}
    }
    (&5).foo(); // Don't lint. `5` will call `<i32 as Foo>::foo`
    (&&5).foo();
    //~^ needless_borrow

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

    // issue #11786
    let x: (&str,) = (&"",);
    //~^ needless_borrow
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
            //~^ needless_borrow
        }
    }

    impl<T, F> S<F>
    where
        F: FnMut() -> T,
    {
        fn calls_mut_field(&mut self) -> T {
            (&mut self.f)()
            //~^ needless_borrow
        }
    }
}

fn issue9383() {
    // Should not lint because unions need explicit deref when accessing field
    use std::mem::ManuallyDrop;

    #[derive(Clone, Copy)]
    struct Wrap<T>(T);
    impl<T> core::ops::Deref for Wrap<T> {
        type Target = T;
        fn deref(&self) -> &T {
            &self.0
        }
    }
    impl<T> core::ops::DerefMut for Wrap<T> {
        fn deref_mut(&mut self) -> &mut T {
            &mut self.0
        }
    }

    union U<T: Copy> {
        u: T,
    }

    #[derive(Clone, Copy)]
    struct Foo {
        x: u32,
    }

    unsafe {
        let mut x = U {
            u: ManuallyDrop::new(Foo { x: 0 }),
        };
        let _ = &mut (&mut x.u).x;
        let _ = &mut (&mut { x.u }).x;
        //~^ needless_borrow
        let _ = &mut ({ &mut x.u }).x;

        let mut x = U {
            u: Wrap(ManuallyDrop::new(Foo { x: 0 })),
        };
        let _ = &mut (&mut x.u).x;
        let _ = &mut (&mut { x.u }).x;
        //~^ needless_borrow
        let _ = &mut ({ &mut x.u }).x;

        let mut x = U { u: Wrap(Foo { x: 0 }) };
        let _ = &mut (&mut x.u).x;
        //~^ needless_borrow
        let _ = &mut (&mut { x.u }).x;
        //~^ needless_borrow
        let _ = &mut ({ &mut x.u }).x;
    }
}

mod issue_10253 {
    struct S;
    trait X {
        fn f<T>(&self);
    }
    impl X for &S {
        fn f<T>(&self) {}
    }
    fn f() {
        (&S).f::<()>();
    }
}

fn issue_12268() {
    let option = Some((&1,));
    let x = (&1,);
    option.unwrap_or((&x.0,));
    //~^ needless_borrow

    // compiler
}

fn issue_14743<T>(slice: &[T]) {
    let _ = (&slice).len();
    //~^ needless_borrow

    let slice = slice as *const [T];
    let _ = unsafe { (&*slice).len() };

    // Check that rustc would actually warn if Clippy had suggested removing the reference
    #[expect(dangerous_implicit_autorefs)]
    let _ = unsafe { (*slice).len() };
}
