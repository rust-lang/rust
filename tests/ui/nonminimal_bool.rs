#![feature(lint_reasons)]
#![allow(unused, clippy::diverging_sub_expression, clippy::needless_if)]
#![warn(clippy::nonminimal_bool)]
#![allow(clippy::useless_vec)]

fn main() {
    let a: bool = unimplemented!();
    let b: bool = unimplemented!();
    let c: bool = unimplemented!();
    let d: bool = unimplemented!();
    let e: bool = unimplemented!();
    let _ = !true;
    let _ = !false;
    let _ = !!a;
    let _ = false || a;
    // don't lint on cfgs
    let _ = cfg!(you_shall_not_not_pass) && a;
    let _ = a || !b || !c || !d || !e;
    let _ = !(!a && b);
    let _ = !(!a || b);
    let _ = !a && !(b && c);
}

fn equality_stuff() {
    let a: i32 = unimplemented!();
    let b: i32 = unimplemented!();
    let c: i32 = unimplemented!();
    let d: i32 = unimplemented!();
    let _ = a == b && c == 5 && a == b;
    let _ = a == b || c == 5 || a == b;
    let _ = a == b && c == 5 && b == a;
    let _ = a != b || !(a != b || c == d);
    let _ = a != b && !(a != b && c == d);
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
