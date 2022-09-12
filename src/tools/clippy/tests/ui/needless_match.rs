// run-rustfix
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
        0 => 0,
        1 => 1,
        2 => 2,
        _ => i,
    };
    let s = "test";
    let _: &str = match s {
        "a" => "a",
        "b" => "b",
        s => s,
    };
}

fn custom_type_match() {
    let se = Simple::A;
    let _: Simple = match se {
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
        Ok(a) => Ok(a),
        Err(err) => Err(err),
    };
    let _: Result<i32, i32> = match func_ret_err(0_i32) {
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
    let _: Result<i32, i32> = if let Ok(val) = x { Ok(val) } else { x };
    // Input type mismatch, don't trigger
    #[allow(clippy::question_mark)]
    let _: Result<i32, i32> = if let Err(e) = Ok(1) { Err(e) } else { x };
}

fn if_let_custom_enum(x: Simple) {
    let _: Simple = if let Simple::A = x {
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
            _ if some_bool => e,
            _ => e,
        };

        // should lint
        let _ = match e {
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

fn main() {}
