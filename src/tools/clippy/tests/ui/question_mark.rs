#![feature(try_blocks)]
#![allow(unreachable_code)]
#![allow(dead_code)]
#![allow(clippy::unnecessary_wraps)]

use std::sync::MutexGuard;

fn some_func(a: Option<u32>) -> Option<u32> {
    if a.is_none() {
        //~^ question_mark
        return None;
    }

    a
}

fn some_other_func(a: Option<u32>) -> Option<u32> {
    if a.is_none() {
        return None;
    } else {
        return Some(0);
    }
    unreachable!()
}

pub enum SeemsOption<T> {
    Some(T),
    None,
}

impl<T> SeemsOption<T> {
    pub fn is_none(&self) -> bool {
        match *self {
            SeemsOption::None => true,
            SeemsOption::Some(_) => false,
        }
    }
}

fn returns_something_similar_to_option(a: SeemsOption<u32>) -> SeemsOption<u32> {
    if a.is_none() {
        return SeemsOption::None;
    }

    a
}

pub struct CopyStruct {
    pub opt: Option<u32>,
}

impl CopyStruct {
    #[rustfmt::skip]
    pub fn func(&self) -> Option<u32> {
        if (self.opt).is_none() {
        //~^ question_mark
            return None;
        }

        if self.opt.is_none() {
        //~^ question_mark
            return None
        }

        let _ = if self.opt.is_none() {
        //~^ question_mark
            return None;
        } else {
            self.opt
        };

        let _ = if let Some(x) = self.opt {
        //~^ question_mark
            x
        } else {
            return None;
        };

        self.opt
    }
}

#[derive(Clone)]
pub struct MoveStruct {
    pub opt: Option<Vec<u32>>,
}

impl MoveStruct {
    pub fn ref_func(&self) -> Option<Vec<u32>> {
        if self.opt.is_none() {
            //~^ question_mark
            return None;
        }

        self.opt.clone()
    }

    pub fn mov_func_reuse(self) -> Option<Vec<u32>> {
        if self.opt.is_none() {
            //~^ question_mark
            return None;
        }

        self.opt
    }

    pub fn mov_func_no_use(self) -> Option<Vec<u32>> {
        if self.opt.is_none() {
            //~^ question_mark
            return None;
        }
        Some(Vec::new())
    }

    pub fn if_let_ref_func(self) -> Option<Vec<u32>> {
        let v: &Vec<_> = if let Some(ref v) = self.opt {
            //~^ question_mark
            v
        } else {
            return None;
        };

        Some(v.clone())
    }

    pub fn if_let_mov_func(self) -> Option<Vec<u32>> {
        let v = if let Some(v) = self.opt {
            //~^ question_mark
            v
        } else {
            return None;
        };

        Some(v)
    }
}

fn func() -> Option<i32> {
    macro_rules! opt_none {
        () => {
            None
        };
    }

    fn f() -> Option<String> {
        Some(String::new())
    }

    if f().is_none() {
        //~^ question_mark
        return None;
    }

    let _val = match f() {
        //~^ question_mark
        Some(val) => val,
        None => return None,
    };

    let s: &str = match &Some(String::new()) {
        Some(v) => v,
        None => return None,
    };

    match f() {
        //~^ question_mark
        Some(val) => val,
        None => return None,
    };

    match opt_none!() {
        //~^ question_mark
        Some(x) => x,
        None => return None,
    };

    match f() {
        Some(x) => x,
        None => return opt_none!(),
    };

    match f() {
        Some(val) => {
            println!("{val}");
            val
        },
        None => return None,
    };

    Some(0)
}

fn func_returning_result() -> Result<i32, i32> {
    Ok(1)
}

fn result_func(x: Result<i32, i32>) -> Result<i32, i32> {
    let _ = if let Ok(x) = x { x } else { return x };
    //~^ question_mark

    if x.is_err() {
        //~^ question_mark
        return x;
    }

    let _val = match func_returning_result() {
        //~^ question_mark
        Ok(val) => val,
        Err(err) => return Err(err),
    };

    match func_returning_result() {
        //~^ question_mark
        Ok(val) => val,
        Err(err) => return Err(err),
    };

    // No warning
    let y = if let Ok(x) = x {
        x
    } else {
        return Err(0);
    };

    // issue #7859
    // no warning
    let _ = if let Ok(x) = func_returning_result() {
        x
    } else {
        return Err(0);
    };

    // no warning
    if func_returning_result().is_err() {
        return func_returning_result();
    }

    // no warning
    let _ = if let Err(e) = x { Err(e) } else { Ok(0) };

    // issue #11283
    // no warning
    #[warn(clippy::question_mark_used)]
    {
        if let Err(err) = Ok(()) {
            return Err(err);
        }

        if Err::<i32, _>(0).is_err() {
            return Err(0);
        } else {
            return Ok(0);
        }

        unreachable!()
    }

    Ok(y)
}

fn infer_check() {
    let closure = |x: Result<u8, ()>| {
        // `?` would fail here, as it expands to `Err(val.into())` which is not constrained.
        let _val = match x {
            Ok(val) => val,
            Err(val) => return Err(val),
        };

        Ok(())
    };

    let closure = |x: Result<u8, ()>| -> Result<(), _> {
        // `?` would fail here, as it expands to `Err(val.into())` which is not constrained.
        let _val = match x {
            Ok(val) => val,
            Err(val) => return Err(val),
        };

        Ok(())
    };
}

// see issue #8019
pub enum NotOption {
    None,
    First,
    AfterFirst,
}

fn obj(_: i32) -> Result<(), NotOption> {
    Err(NotOption::First)
}

fn f() -> NotOption {
    if obj(2).is_err() {
        return NotOption::None;
    }
    NotOption::First
}

fn do_something() {}

fn err_immediate_return() -> Result<i32, i32> {
    if let Err(err) = func_returning_result() {
        //~^ question_mark
        return Err(err);
    }
    Ok(1)
}

fn err_immediate_return_and_do_something() -> Result<i32, i32> {
    if let Err(err) = func_returning_result() {
        //~^ question_mark
        return Err(err);
    }
    do_something();
    Ok(1)
}

// No warning
fn no_immediate_return() -> Result<i32, i32> {
    if let Err(err) = func_returning_result() {
        do_something();
        return Err(err);
    }
    Ok(1)
}

// No warning
fn mixed_result_and_option() -> Option<i32> {
    if let Err(err) = func_returning_result() {
        return Some(err);
    }
    None
}

// No warning
fn else_if_check() -> Result<i32, i32> {
    if true {
        Ok(1)
    } else if let Err(e) = func_returning_result() {
        Err(e)
    } else {
        Err(-1)
    }
}

// No warning
#[allow(clippy::manual_map)]
#[rustfmt::skip]
fn option_map() -> Option<bool> {
    if let Some(a) = Some(false) {
        Some(!a)
    } else {
        None
    }
}

pub struct PatternedError {
    flag: bool,
}

// No warning
fn pattern() -> Result<(), PatternedError> {
    let res = Ok(());

    if let Err(err @ PatternedError { flag: true }) = res {
        return Err(err);
    }

    res
}

fn main() {}

// `?` is not the same as `return None;` if inside of a try block
fn issue8628(a: Option<u32>) -> Option<u32> {
    let b: Option<u32> = try {
        if a.is_none() {
            return None;
        }
        32
    };
    b.or(Some(128))
}

fn issue6828_nested_body() -> Option<u32> {
    try {
        fn f2(a: Option<i32>) -> Option<i32> {
            if a.is_none() {
                //~^ question_mark
                return None;
                // do lint here, the outer `try` is not relevant here
                // https://github.com/rust-lang/rust-clippy/pull/11001#issuecomment-1610636867
            }
            Some(32)
        }
        123
    }
}

// should not lint, `?` operator not available in const context
const fn issue9175(option: Option<()>) -> Option<()> {
    if option.is_none() {
        return None;
    }
    //stuff
    Some(())
}

fn issue12337() -> Option<i32> {
    let _: Option<i32> = try {
        let Some(_) = Some(42) else {
            return None;
        };
        123
    };
    Some(42)
}

fn issue11983(option: &Option<String>) -> Option<()> {
    // Don't lint, `&Option` dose not impl `Try`.
    let Some(v) = option else { return None };

    let opt = Some(String::new());
    // Don't lint, `branch` method in `Try` takes ownership of `opt`,
    // and `(&opt)?` also doesn't work since it's `&Option`.
    let Some(v) = &opt else { return None };
    let mov = opt;

    Some(())
}

struct Foo {
    owned: Option<String>,
}
struct Bar {
    foo: Foo,
}
#[allow(clippy::disallowed_names)]
fn issue12412(foo: &Foo, bar: &Bar) -> Option<()> {
    // Don't lint, `owned` is behind a shared reference.
    let Some(v) = &foo.owned else {
        return None;
    };
    // Don't lint, `owned` is behind a shared reference.
    let Some(v) = &bar.foo.owned else {
        return None;
    };
    // lint
    let Some(v) = bar.foo.owned.clone() else {
        return None;
    };
    //~^^^ question_mark
    Some(())
}

struct StructWithOptionString {
    opt_x: Option<String>,
}

struct WrapperStructWithString(String);

#[allow(clippy::disallowed_names)]
fn issue_13417(foo: &mut StructWithOptionString) -> Option<String> {
    let Some(ref x) = foo.opt_x else {
        return None;
    };
    //~^^^ question_mark
    let opt_y = Some(x.clone());
    std::mem::replace(&mut foo.opt_x, opt_y)
}

#[allow(clippy::disallowed_names)]
fn issue_13417_mut(foo: &mut StructWithOptionString) -> Option<String> {
    let Some(ref mut x) = foo.opt_x else {
        return None;
    };
    //~^^^ question_mark
    let opt_y = Some(x.clone());
    std::mem::replace(&mut foo.opt_x, opt_y)
}

#[allow(clippy::disallowed_names)]
#[allow(unused)]
fn issue_13417_weirder(foo: &mut StructWithOptionString, mut bar: Option<WrapperStructWithString>) -> Option<()> {
    let Some(ref x @ ref y) = foo.opt_x else {
        return None;
    };
    //~^^^ question_mark
    let Some(ref x @ WrapperStructWithString(_)) = bar else {
        return None;
    };
    //~^^^ question_mark
    let Some(ref mut x @ WrapperStructWithString(_)) = bar else {
        return None;
    };
    //~^^^ question_mark
    Some(())
}

#[clippy::msrv = "1.12"]
fn msrv_1_12(arg: Option<i32>) -> Option<i32> {
    if arg.is_none() {
        return None;
    }
    let val = match arg {
        Some(val) => val,
        None => return None,
    };
    println!("{}", val);
    Some(val)
}

#[clippy::msrv = "1.13"]
fn msrv_1_13(arg: Option<i32>) -> Option<i32> {
    if arg.is_none() {
        //~^ question_mark
        return None;
    }
    let val = match arg {
        //~^ question_mark
        Some(val) => val,
        None => return None,
    };
    println!("{}", val);
    Some(val)
}

fn issue_14615(a: MutexGuard<Option<u32>>) -> Option<String> {
    let Some(a) = *a else {
        return None;
    };
    //~^^^ question_mark
    Some(format!("{a}"))
}
