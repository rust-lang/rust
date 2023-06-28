//@run-rustfix
#![feature(try_blocks)]
#![allow(unreachable_code)]
#![allow(dead_code)]
#![allow(clippy::unnecessary_wraps)]

fn some_func(a: Option<u32>) -> Option<u32> {
    if a.is_none() {
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
            return None;
        }

        if self.opt.is_none() {
            return None
        }

        let _ = if self.opt.is_none() {
            return None;
        } else {
            self.opt
        };

        let _ = if let Some(x) = self.opt {
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
            return None;
        }

        self.opt.clone()
    }

    pub fn mov_func_reuse(self) -> Option<Vec<u32>> {
        if self.opt.is_none() {
            return None;
        }

        self.opt
    }

    pub fn mov_func_no_use(self) -> Option<Vec<u32>> {
        if self.opt.is_none() {
            return None;
        }
        Some(Vec::new())
    }

    pub fn if_let_ref_func(self) -> Option<Vec<u32>> {
        let v: &Vec<_> = if let Some(ref v) = self.opt {
            v
        } else {
            return None;
        };

        Some(v.clone())
    }

    pub fn if_let_mov_func(self) -> Option<Vec<u32>> {
        let v = if let Some(v) = self.opt {
            v
        } else {
            return None;
        };

        Some(v)
    }
}

fn func() -> Option<i32> {
    fn f() -> Option<String> {
        Some(String::new())
    }

    if f().is_none() {
        return None;
    }

    Some(0)
}

fn func_returning_result() -> Result<i32, i32> {
    Ok(1)
}

fn result_func(x: Result<i32, i32>) -> Result<i32, i32> {
    let _ = if let Ok(x) = x { x } else { return x };

    if x.is_err() {
        return x;
    }

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

    Ok(y)
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
        return Err(err);
    }
    Ok(1)
}

fn err_immediate_return_and_do_something() -> Result<i32, i32> {
    if let Err(err) = func_returning_result() {
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
