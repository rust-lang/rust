//@aux-build:proc_macros.rs
#![feature(if_let_guard)]
#![allow(clippy::no_effect, unused, clippy::single_match, invalid_nan_comparisons)]
#![warn(clippy::redundant_guards)]

#[macro_use]
extern crate proc_macros;

struct A(u32);

struct B {
    e: Option<A>,
}

struct C(u32, u32);

#[derive(PartialEq)]
struct FloatWrapper(f32);

fn issue11304() {
    match 0.1 {
        x if x == 0.0 => todo!(),
        //~^ redundant_guards
        // Pattern matching NAN is illegal
        x if x == f64::NAN => todo!(),
        _ => todo!(),
    }
    match FloatWrapper(0.1) {
        x if x == FloatWrapper(0.0) => todo!(),
        //~^ redundant_guards
        _ => todo!(),
    }
}

fn issue13681() {
    match c"hi" {
        x if x == c"hi" => (),
        _ => (),
    }
}

fn main() {
    let c = C(1, 2);
    match c {
        C(x, y) if let 1 = y => ..,
        //~^ redundant_guards
        _ => todo!(),
    };

    let x = Some(Some(1));
    match x {
        Some(x) if matches!(x, Some(1) if true) => ..,
        //~^ redundant_guards
        Some(x) if matches!(x, Some(1)) => {
            //~^ redundant_guards
            println!("a");
            ..
        },
        Some(x) if let Some(1) = x => ..,
        //~^ redundant_guards
        Some(x) if x == Some(2) => ..,
        //~^ redundant_guards
        Some(x) if Some(2) == x => ..,
        //~^ redundant_guards
        // Don't lint, since x is used in the body
        Some(x) if let Some(1) = x => {
            x;
            ..
        },
        _ => todo!(),
    };
    let y = 1;
    match x {
        // Don't inline these, since y is not from the pat
        Some(x) if matches!(y, 1 if true) => ..,
        Some(x) if let 1 = y => ..,
        Some(x) if y == 2 => ..,
        Some(x) if 2 == y => ..,
        _ => todo!(),
    };
    let a = A(1);
    match a {
        _ if a.0 == 1 => {},
        _ if 1 == a.0 => {},
        _ => todo!(),
    }
    let b = B { e: Some(A(0)) };
    match b {
        B { e } if matches!(e, Some(A(2))) => ..,
        //~^ redundant_guards
        _ => todo!(),
    };
    // Do not lint, since we cannot represent this as a pattern (at least, without a conversion)
    let v = Some(vec![1u8, 2, 3]);
    match v {
        Some(x) if x == [1] => {},
        _ => {},
    }

    external! {
        let x = Some(Some(1));
        match x {
            Some(x) if let Some(1) = x => ..,
            _ => todo!(),
        };
    }
    with_span! {
        span
        let x = Some(Some(1));
        match x {
            Some(x) if let Some(1) = x => ..,
            _ => todo!(),
        };
    }
}

enum E {
    A(&'static str),
    B(&'static str),
    C(&'static str),
}

fn i() {
    match E::A("") {
        // Do not lint
        E::A(x) | E::B(x) | E::C(x) if x == "from an or pattern" => {},
        E::A(y) if y == "not from an or pattern" => {},
        //~^ redundant_guards
        _ => {},
    };
}

fn h(v: Option<u32>) {
    match v {
        x if matches!(x, Some(0)) => ..,
        //~^ redundant_guards
        _ => ..,
    };
}

fn negative_literal(i: i32) {
    match i {
        i if i == -1 => {},
        //~^ redundant_guards
        i if i == 1 => {},
        //~^ redundant_guards
        _ => {},
    }
}

// Do not lint

fn f(s: Option<std::ffi::OsString>) {
    match s {
        Some(x) if x == "a" => {},
        Some(x) if "a" == x => {},
        _ => {},
    }
}

fn not_matches() {
    match Some(42) {
        // The pattern + guard is not equivalent to `Some(42)` because of the `panic!`
        Some(v)
            if match v {
                42 => true,
                _ => panic!(),
            } => {},
        _ => {},
    }
}

struct S {
    a: usize,
}

impl PartialEq for S {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for S {}

static CONST_S: S = S { a: 1 };

fn g(opt_s: Option<S>) {
    match opt_s {
        Some(x) if x == CONST_S => {},
        Some(x) if CONST_S == x => {},
        _ => {},
    }
}

mod issue11465 {
    enum A {
        Foo([u8; 3]),
    }

    struct B {
        b: String,
        c: i32,
    }

    fn issue11465() {
        let c = Some(1);
        match c {
            Some(ref x) if x == &1 => {},
            //~^ redundant_guards
            Some(ref x) if &1 == x => {},
            //~^ redundant_guards
            Some(ref x) if let &2 = x => {},
            //~^ redundant_guards
            Some(ref x) if matches!(x, &3) => {},
            //~^ redundant_guards
            _ => {},
        };

        let enum_a = A::Foo([98, 97, 114]);
        match enum_a {
            A::Foo(ref arr) if arr == b"foo" => {},
            A::Foo(ref arr) if b"foo" == arr => {},
            A::Foo(ref arr) if let b"bar" = arr => {},
            A::Foo(ref arr) if matches!(arr, b"baz") => {},
            _ => {},
        };

        let struct_b = B {
            b: "bar".to_string(),
            c: 42,
        };
        match struct_b {
            B { ref b, .. } if b == "bar" => {},
            B { ref b, .. } if "bar" == b => {},
            B { ref c, .. } if c == &1 => {},
            //~^ redundant_guards
            B { ref c, .. } if &1 == c => {},
            //~^ redundant_guards
            B { ref c, .. } if let &1 = c => {},
            //~^ redundant_guards
            B { ref c, .. } if matches!(c, &1) => {},
            //~^ redundant_guards
            _ => {},
        }
    }
}

fn issue11807() {
    #![allow(clippy::single_match)]

    match Some(Some("")) {
        Some(Some(x)) if x.is_empty() => {},
        //~^ redundant_guards
        _ => {},
    }

    match Some(Some(String::new())) {
        // Do not lint: String deref-coerces to &str
        Some(Some(x)) if x.is_empty() => {},
        _ => {},
    }

    match Some(Some(&[] as &[i32])) {
        Some(Some(x)) if x.is_empty() => {},
        //~^ redundant_guards
        _ => {},
    }

    match Some(Some([] as [i32; 0])) {
        Some(Some(x)) if x.is_empty() => {},
        //~^ redundant_guards
        _ => {},
    }

    match Some(Some(Vec::<()>::new())) {
        // Do not lint: Vec deref-coerces to &[T]
        Some(Some(x)) if x.is_empty() => {},
        _ => {},
    }

    match Some(Some(&[] as &[i32])) {
        Some(Some(x)) if x.starts_with(&[]) => {},
        //~^ redundant_guards
        _ => {},
    }

    match Some(Some(&[] as &[i32])) {
        Some(Some(x)) if x.starts_with(&[1]) => {},
        //~^ redundant_guards
        _ => {},
    }

    match Some(Some(&[] as &[i32])) {
        Some(Some(x)) if x.starts_with(&[1, 2]) => {},
        //~^ redundant_guards
        _ => {},
    }

    match Some(Some(&[] as &[i32])) {
        Some(Some(x)) if x.ends_with(&[1, 2]) => {},
        //~^ redundant_guards
        _ => {},
    }

    match Some(Some(Vec::<i32>::new())) {
        // Do not lint: deref coercion
        Some(Some(x)) if x.starts_with(&[1, 2]) => {},
        _ => {},
    }
}

mod issue12243 {
    pub const fn const_fn(x: &str) {
        match x {
            // Shouldn't lint.
            y if y.is_empty() => {},
            _ => {},
        }
    }

    pub fn non_const_fn(x: &str) {
        match x {
            y if y.is_empty() => {},
            //~^ ERROR: redundant guard
            _ => {},
        }
    }

    struct Bar;

    impl Bar {
        pub const fn const_bar(x: &str) {
            match x {
                // Shouldn't lint.
                y if y.is_empty() => {},
                _ => {},
            }
        }

        pub fn non_const_bar(x: &str) {
            match x {
                y if y.is_empty() => {},
                //~^ ERROR: redundant guard
                _ => {},
            }
        }
    }

    static FOO: () = {
        match "" {
            // Shouldn't lint.
            x if x.is_empty() => {},
            _ => {},
        }
    };
}
