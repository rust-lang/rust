//@no-rustfix: overlapping suggestions
#![allow(
    unused,
    clippy::diverging_sub_expression,
    clippy::needless_if,
    clippy::redundant_pattern_matching
)]
#![warn(clippy::nonminimal_bool)]
#![allow(clippy::useless_vec)]

fn main() {
    let a: bool = unimplemented!();
    let b: bool = unimplemented!();
    let c: bool = unimplemented!();
    let d: bool = unimplemented!();
    let e: bool = unimplemented!();
    let _ = !true;
    //~^ nonminimal_bool

    let _ = !false;
    //~^ nonminimal_bool

    let _ = !!a;
    //~^ nonminimal_bool

    let _ = false || a;
    //~^ nonminimal_bool

    // don't lint on cfgs
    let _ = cfg!(you_shall_not_not_pass) && a;
    let _ = a || !b || !c || !d || !e;
    let _ = !(!a && b);
    //~^ nonminimal_bool

    let _ = !(!a || b);
    //~^ nonminimal_bool

    let _ = !a && !(b && c);
    //~^ nonminimal_bool
}

fn equality_stuff() {
    let a: i32 = unimplemented!();
    let b: i32 = unimplemented!();
    let c: i32 = unimplemented!();
    let d: i32 = unimplemented!();
    let _ = a == b && c == 5 && a == b;
    //~^ nonminimal_bool

    let _ = a == b || c == 5 || a == b;
    //~^ nonminimal_bool

    let _ = a == b && c == 5 && b == a;
    //~^ nonminimal_bool

    let _ = a != b || !(a != b || c == d);
    //~^ nonminimal_bool

    let _ = a != b && !(a != b && c == d);
    //~^ nonminimal_bool
}

fn issue3847(a: u32, b: u32) -> bool {
    const THRESHOLD: u32 = 1_000;

    if a < THRESHOLD && b >= THRESHOLD || a >= THRESHOLD && b < THRESHOLD {
        return false;
    }
    true
}

fn issue4548() {
    fn f(_i: u32, _j: u32) -> u32 {
        unimplemented!();
    }

    let i = 0;
    let j = 0;

    if i != j && f(i, j) != 0 || i == j && f(i, j) != 1 {}
}

fn check_expect() {
    let a: bool = unimplemented!();
    #[expect(clippy::nonminimal_bool)]
    let _ = !!a;
}

fn issue9428() {
    if matches!(true, true) && true {
        //~^ nonminimal_bool

        println!("foo");
    }
}

fn issue_10523() {
    macro_rules! a {
        ($v:expr) => {
            $v.is_some()
        };
    }
    let x: Option<u32> = None;
    if !a!(x) {}
}

fn issue_10523_1() {
    macro_rules! a {
        ($v:expr) => {
            !$v.is_some()
        };
    }
    let x: Option<u32> = None;
    if a!(x) {}
}

fn issue_10523_2() {
    macro_rules! a {
        () => {
            !None::<u32>.is_some()
        };
    }
    if a!() {}
}

fn issue_10435() {
    let x = vec![0];
    let y = vec![1];
    let z = vec![2];

    // vvv Should not lint
    #[allow(clippy::nonminimal_bool)]
    if !x.is_empty() && !(y.is_empty() || z.is_empty()) {
        println!("{}", line!());
    }

    // vvv Should not lint (#10435 talks about a bug where it lints)
    #[allow(clippy::nonminimal_bool)]
    if !(x == [0]) {
        println!("{}", line!());
    }
}

fn issue10836() {
    struct Foo(bool);
    impl std::ops::Not for Foo {
        type Output = bool;

        fn not(self) -> Self::Output {
            !self.0
        }
    }

    // Should not lint
    let _: bool = !!Foo(true);
}

fn issue11932() {
    let x: i32 = unimplemented!();

    #[allow(clippy::nonminimal_bool)]
    let _ = x % 2 == 0 || {
        // Should not lint
        assert!(x > 0);
        x % 3 == 0
    };
}

fn issue_5794() {
    let a = 0;
    if !(12 == a) {}
    //~^ nonminimal_bool
    if !(a == 12) {}
    //~^ nonminimal_bool
    if !(12 != a) {}
    //~^ nonminimal_bool
    if !(a != 12) {}
    //~^ nonminimal_bool

    let b = true;
    let c = false;
    if !b == true {}
    //~^ nonminimal_bool
    //~| bool_comparison
    //~| bool_comparison
    if !b != true {}
    //~^ nonminimal_bool
    //~| bool_comparison
    if true == !b {}
    //~^ nonminimal_bool
    //~| bool_comparison
    //~| bool_comparison
    if true != !b {}
    //~^ nonminimal_bool
    //~| bool_comparison
    if !b == !c {}
    //~^ nonminimal_bool
    if !b != !c {}
    //~^ nonminimal_bool
}

fn issue_12371(x: usize) -> bool {
    // Should not warn!
    !x != 0
}

// Not linted because it is slow to do so
// https://github.com/rust-lang/rust-clippy/issues/13206
fn many_ops(a: bool, b: bool, c: bool, d: bool, e: bool, f: bool) -> bool {
    (a && c && f) || (!a && b && !d) || (!b && !c && !e) || (d && e && !f)
}

fn issue14184(a: f32, b: bool) {
    if !(a < 2.0 && !b) {
        //~^ nonminimal_bool
        println!("Hi");
    }
}

mod issue14404 {
    enum TyKind {
        Ref(i32, i32, i32),
        Other,
    }

    struct Expr;

    fn is_mutable(expr: &Expr) -> bool {
        todo!()
    }

    fn should_not_give_macro(ty: TyKind, expr: Expr) {
        if !(matches!(ty, TyKind::Ref(_, _, _)) && !is_mutable(&expr)) {
            //~^ nonminimal_bool
            todo!()
        }
    }
}
