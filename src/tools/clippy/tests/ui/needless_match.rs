#![warn(clippy::needless_match)]
#![allow(clippy::manual_map)]
#![allow(dead_code)]

#[derive(Clone, Copy)]
enum Simple {
    A,
    B,
    C,
    D,
}

fn useless_match() {
    let i = 10;
    let _: i32 = match i {
        //~^ needless_match
        0 => 0,
        1 => 1,
        2 => 2,
        _ => i,
    };
    let s = "test";
    let _: &str = match s {
        //~^ needless_match
        "a" => "a",
        "b" => "b",
        s => s,
    };
}

fn custom_type_match() {
    let se = Simple::A;
    let _: Simple = match se {
        //~^ needless_match
        Simple::A => Simple::A,
        Simple::B => Simple::B,
        Simple::C => Simple::C,
        Simple::D => Simple::D,
    };
    // Don't trigger
    let _: Simple = match se {
        Simple::A => Simple::A,
        Simple::B => Simple::B,
        _ => Simple::C,
    };
    // Mingled, don't trigger
    let _: Simple = match se {
        Simple::A => Simple::B,
        Simple::B => Simple::C,
        Simple::C => Simple::D,
        Simple::D => Simple::A,
    };
}

fn option_match(x: Option<i32>) {
    let _: Option<i32> = match x {
        //~^ needless_match
        Some(a) => Some(a),
        None => None,
    };
    // Don't trigger, this is the case for manual_map_option
    let _: Option<i32> = match x {
        Some(a) => Some(-a),
        None => None,
    };
}

fn func_ret_err<T>(err: T) -> Result<i32, T> {
    Err(err)
}

fn result_match() {
    let _: Result<i32, i32> = match Ok(1) {
        //~^ needless_match
        Ok(a) => Ok(a),
        Err(err) => Err(err),
    };
    let _: Result<i32, i32> = match func_ret_err(0_i32) {
        //~^ needless_match
        Err(err) => Err(err),
        Ok(a) => Ok(a),
    };
    // as ref, don't trigger
    let res = &func_ret_err(0_i32);
    let _: Result<&i32, &i32> = match *res {
        Ok(ref x) => Ok(x),
        Err(ref x) => Err(x),
    };
}

fn if_let_option() {
    let _ = if let Some(a) = Some(1) { Some(a) } else { None };
    //~^ needless_match

    fn do_something() {}

    // Don't trigger
    let _ = if let Some(a) = Some(1) {
        Some(a)
    } else {
        do_something();
        None
    };

    // Don't trigger
    let _ = if let Some(a) = Some(1) {
        do_something();
        Some(a)
    } else {
        None
    };

    // Don't trigger
    let _ = if let Some(a) = Some(1) { Some(a) } else { Some(2) };
}

fn if_let_option_result() -> Result<(), ()> {
    fn f(x: i32) -> Result<Option<i32>, ()> {
        Ok(Some(x))
    }
    // Don't trigger
    let _ = if let Some(v) = f(1)? { Some(v) } else { f(2)? };
    Ok(())
}

fn if_let_result() {
    let x: Result<i32, i32> = Ok(1);
    let _: Result<i32, i32> = if let Err(e) = x { Err(e) } else { x };
    //~^ needless_match
    let _: Result<i32, i32> = if let Ok(val) = x { Ok(val) } else { x };
    //~^ needless_match
    // Input type mismatch, don't trigger
    #[allow(clippy::question_mark)]
    let _: Result<i32, i32> = if let Err(e) = Ok(1) { Err(e) } else { x };
}

fn if_let_custom_enum(x: Simple) {
    let _: Simple = if let Simple::A = x {
        //~^ needless_match
        Simple::A
    } else if let Simple::B = x {
        Simple::B
    } else if let Simple::C = x {
        Simple::C
    } else {
        x
    };

    // Don't trigger
    let _: Simple = if let Simple::A = x {
        Simple::A
    } else if true {
        Simple::B
    } else {
        x
    };
}

mod issue8542 {
    #[derive(Clone, Copy)]
    enum E {
        VariantA(u8, u8),
        VariantB(u8, bool),
    }

    enum Complex {
        A(u8),
        B(u8, bool),
        C(u8, i32, f64),
        D(E, bool),
    }

    fn match_test() {
        let ce = Complex::B(8, false);
        let aa = 0_u8;
        let bb = false;

        let _: Complex = match ce {
            //~^ needless_match
            Complex::A(a) => Complex::A(a),
            Complex::B(a, b) => Complex::B(a, b),
            Complex::C(a, b, c) => Complex::C(a, b, c),
            Complex::D(E::VariantA(ea, eb), b) => Complex::D(E::VariantA(ea, eb), b),
            Complex::D(E::VariantB(ea, eb), b) => Complex::D(E::VariantB(ea, eb), b),
        };

        // Don't trigger
        let _: Complex = match ce {
            Complex::A(_) => Complex::A(aa),
            Complex::B(_, b) => Complex::B(aa, b),
            Complex::C(_, b, _) => Complex::C(aa, b, 64_f64),
            Complex::D(e, b) => Complex::D(e, b),
        };

        // Don't trigger
        let _: Complex = match ce {
            Complex::A(a) => Complex::A(a),
            Complex::B(a, _) => Complex::B(a, bb),
            Complex::C(a, _, _) => Complex::C(a, 32_i32, 64_f64),
            _ => ce,
        };
    }
}

/// Lint triggered when type coercions happen.
/// Do NOT trigger on any of these.
mod issue8551 {
    trait Trait {}
    struct Struct;
    impl Trait for Struct {}

    fn optmap(s: Option<&Struct>) -> Option<&dyn Trait> {
        match s {
            Some(s) => Some(s),
            None => None,
        }
    }

    fn lint_tests() {
        let option: Option<&Struct> = None;
        let _: Option<&dyn Trait> = match option {
            Some(s) => Some(s),
            None => None,
        };

        let _: Option<&dyn Trait> = if true {
            match option {
                Some(s) => Some(s),
                None => None,
            }
        } else {
            None
        };

        let result: Result<&Struct, i32> = Err(0);
        let _: Result<&dyn Trait, i32> = match result {
            Ok(s) => Ok(s),
            Err(e) => Err(e),
        };

        let _: Option<&dyn Trait> = if let Some(s) = option { Some(s) } else { None };
    }
}

trait Tr {
    fn as_mut(&mut self) -> Result<&mut i32, &mut i32>;
}
impl Tr for Result<i32, i32> {
    fn as_mut(&mut self) -> Result<&mut i32, &mut i32> {
        match self {
            Ok(x) => Ok(x),
            Err(e) => Err(e),
        }
    }
}

mod issue9084 {
    fn wildcard_if() {
        let mut some_bool = true;
        let e = Some(1);

        // should lint
        let _ = match e {
            //~^ needless_match
            _ if some_bool => e,
            _ => e,
        };

        // should lint
        let _ = match e {
            //~^ needless_match
            Some(i) => Some(i),
            _ if some_bool => e,
            _ => e,
        };

        // should not lint
        let _ = match e {
            _ if some_bool => e,
            _ => Some(2),
        };

        // should not lint
        let _ = match e {
            Some(i) => Some(i + 1),
            _ if some_bool => e,
            _ => e,
        };

        // should not lint (guard has side effects)
        let _ = match e {
            Some(i) => Some(i),
            _ if {
                some_bool = false;
                some_bool
            } =>
            {
                e
            },
            _ => e,
        };
    }
}

fn a() -> Option<()> {
    Some(())
}
fn b() -> Option<()> {
    Some(())
}
fn c() -> Option<()> {
    Some(())
}

#[allow(clippy::ifs_same_cond)]
pub fn issue13574() -> Option<()> {
    // Do not lint.
    // The right hand of all these arms are different functions.
    let _ = {
        if let Some(a) = a() {
            Some(a)
        } else if let Some(b) = b() {
            Some(b)
        } else if let Some(c) = c() {
            Some(c)
        } else {
            None
        }
    };

    const A: Option<()> = Some(());
    const B: Option<()> = Some(());
    const C: Option<()> = Some(());
    const D: Option<()> = Some(());

    let _ = {
        if let Some(num) = A {
            Some(num)
        } else if let Some(num) = B {
            Some(num)
        } else if let Some(num) = C {
            Some(num)
        } else if let Some(num) = D {
            Some(num)
        } else {
            None
        }
    };

    // Same const, should lint
    let _ = {
        if let Some(num) = A {
            //~^ needless_match
            Some(num)
        } else if let Some(num) = A {
            Some(num)
        } else if let Some(num) = A {
            Some(num)
        } else {
            None
        }
    };

    None
}

fn issue14754(t: Result<i32, &'static str>) -> Result<i32, &'static str> {
    let _ = match t {
        Ok(v) => Ok::<_, &'static str>(v),
        err @ Err(_) => return err,
    };
    println!("Still here");
    let x = match t {
        Ok(v) => Ok::<_, &'static str>(v),
        err @ Err(_) => err,
    };
    //~^^^^ needless_match
    println!("Still here");
    x
}

fn main() {}
