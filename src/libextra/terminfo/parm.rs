// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Parameterized string expansion

use core::prelude::*;
use core::{char, int, vec};
use core::iterator::IteratorUtil;

#[deriving(Eq)]
enum States {
    Nothing,
    Percent,
    SetVar,
    GetVar,
    PushParam,
    CharConstant,
    CharClose,
    IntConstant,
    SeekIfElse(int),
    SeekIfElsePercent(int),
    SeekIfEnd(int),
    SeekIfEndPercent(int)
}

/// Types of parameters a capability can use
pub enum Param {
    String(~str),
    Number(int)
}

/// Container for static and dynamic variable arrays
pub struct Variables {
    /// Static variables A-Z
    sta: [Param, ..26],
    /// Dynamic variables a-z
    dyn: [Param, ..26]
}

impl Variables {
    /// Return a new zero-initialized Variables
    pub fn new() -> Variables {
        Variables{ sta: [Number(0), ..26], dyn: [Number(0), ..26] }
    }
}

/**
  Expand a parameterized capability

  # Arguments
  * `cap`    - string to expand
  * `params` - vector of params for %p1 etc
  * `vars`   - Variables struct for %Pa etc

  To be compatible with ncurses, `vars` should be the same between calls to `expand` for
  multiple capabilities for the same terminal.
  */
pub fn expand(cap: &[u8], params: &[Param], vars: &mut Variables)
    -> Result<~[u8], ~str> {
    let mut state = Nothing;

    // expanded cap will only rarely be larger than the cap itself
    let mut output = vec::with_capacity(cap.len());

    let mut stack: ~[Param] = ~[];

    let mut intstate = ~[];

    // Copy parameters into a local vector for mutability
    let mut mparams = [Number(0), ..9];
    for mparams.mut_iter().zip(params.iter()).advance |(dst, &src)| {
        *dst = src;
    }

    for cap.iter().transform(|&x| x).advance |c| {
        let cur = c as char;
        let mut old_state = state;
        match state {
            Nothing => {
                if cur == '%' {
                    state = Percent;
                } else {
                    output.push(c);
                }
            },
            Percent => {
                match cur {
                    '%' => { output.push(c); state = Nothing },
                    'c' => if stack.len() > 0 {
                        match stack.pop() {
                            // if c is 0, use 0200 (128) for ncurses compatibility
                            Number(c) => output.push(if c == 0 { 128 } else { c } as u8),
                            _       => return Err(~"a non-char was used with %c")
                        }
                    } else { return Err(~"stack is empty") },
                    's' => if stack.len() > 0 {
                        match stack.pop() {
                            String(s) => output.push_all(s.as_bytes()),
                            _         => return Err(~"a non-str was used with %s")
                        }
                    } else { return Err(~"stack is empty") },
                    'd' => if stack.len() > 0 {
                        match stack.pop() {
                            Number(x) => {
                                let s = x.to_str();
                                output.push_all(s.as_bytes())
                            }
                            _         => return Err(~"a non-number was used with %d")
                        }
                    } else { return Err(~"stack is empty") },
                    'p' => state = PushParam,
                    'P' => state = SetVar,
                    'g' => state = GetVar,
                    '\'' => state = CharConstant,
                    '{' => state = IntConstant,
                    'l' => if stack.len() > 0 {
                        match stack.pop() {
                            String(s) => stack.push(Number(s.len() as int)),
                            _         => return Err(~"a non-str was used with %l")
                        }
                    } else { return Err(~"stack is empty") },
                    '+' => if stack.len() > 1 {
                        match (stack.pop(), stack.pop()) {
                            (Number(y), Number(x)) => stack.push(Number(x + y)),
                            _ => return Err(~"non-numbers on stack with +")
                        }
                    } else { return Err(~"stack is empty") },
                    '-' => if stack.len() > 1 {
                        match (stack.pop(), stack.pop()) {
                            (Number(y), Number(x)) => stack.push(Number(x - y)),
                            _ => return Err(~"non-numbers on stack with -")
                        }
                    } else { return Err(~"stack is empty") },
                    '*' => if stack.len() > 1 {
                        match (stack.pop(), stack.pop()) {
                            (Number(y), Number(x)) => stack.push(Number(x * y)),
                            _ => return Err(~"non-numbers on stack with *")
                        }
                    } else { return Err(~"stack is empty") },
                    '/' => if stack.len() > 1 {
                        match (stack.pop(), stack.pop()) {
                            (Number(y), Number(x)) => stack.push(Number(x / y)),
                            _ => return Err(~"non-numbers on stack with /")
                        }
                    } else { return Err(~"stack is empty") },
                    'm' => if stack.len() > 1 {
                        match (stack.pop(), stack.pop()) {
                            (Number(y), Number(x)) => stack.push(Number(x % y)),
                            _ => return Err(~"non-numbers on stack with %")
                        }
                    } else { return Err(~"stack is empty") },
                    '&' => if stack.len() > 1 {
                        match (stack.pop(), stack.pop()) {
                            (Number(y), Number(x)) => stack.push(Number(x & y)),
                            _ => return Err(~"non-numbers on stack with &")
                        }
                    } else { return Err(~"stack is empty") },
                    '|' => if stack.len() > 1 {
                        match (stack.pop(), stack.pop()) {
                            (Number(y), Number(x)) => stack.push(Number(x | y)),
                            _ => return Err(~"non-numbers on stack with |")
                        }
                    } else { return Err(~"stack is empty") },
                    '^' => if stack.len() > 1 {
                        match (stack.pop(), stack.pop()) {
                            (Number(y), Number(x)) => stack.push(Number(x ^ y)),
                            _ => return Err(~"non-numbers on stack with ^")
                        }
                    } else { return Err(~"stack is empty") },
                    '=' => if stack.len() > 1 {
                        match (stack.pop(), stack.pop()) {
                            (Number(y), Number(x)) => stack.push(Number(if x == y { 1 }
                                                                        else { 0 })),
                            _ => return Err(~"non-numbers on stack with =")
                        }
                    } else { return Err(~"stack is empty") },
                    '>' => if stack.len() > 1 {
                        match (stack.pop(), stack.pop()) {
                            (Number(y), Number(x)) => stack.push(Number(if x > y { 1 }
                                                                        else { 0 })),
                            _ => return Err(~"non-numbers on stack with >")
                        }
                    } else { return Err(~"stack is empty") },
                    '<' => if stack.len() > 1 {
                        match (stack.pop(), stack.pop()) {
                            (Number(y), Number(x)) => stack.push(Number(if x < y { 1 }
                                                                        else { 0 })),
                            _ => return Err(~"non-numbers on stack with <")
                        }
                    } else { return Err(~"stack is empty") },
                    'A' => if stack.len() > 1 {
                        match (stack.pop(), stack.pop()) {
                            (Number(0), Number(_)) => stack.push(Number(0)),
                            (Number(_), Number(0)) => stack.push(Number(0)),
                            (Number(_), Number(_)) => stack.push(Number(1)),
                            _ => return Err(~"non-numbers on stack with logical and")
                        }
                    } else { return Err(~"stack is empty") },
                    'O' => if stack.len() > 1 {
                        match (stack.pop(), stack.pop()) {
                            (Number(0), Number(0)) => stack.push(Number(0)),
                            (Number(_), Number(_)) => stack.push(Number(1)),
                            _ => return Err(~"non-numbers on stack with logical or")
                        }
                    } else { return Err(~"stack is empty") },
                    '!' => if stack.len() > 0 {
                        match stack.pop() {
                            Number(0) => stack.push(Number(1)),
                            Number(_) => stack.push(Number(0)),
                            _ => return Err(~"non-number on stack with logical not")
                        }
                    } else { return Err(~"stack is empty") },
                    '~' => if stack.len() > 0 {
                        match stack.pop() {
                            Number(x) => stack.push(Number(!x)),
                            _         => return Err(~"non-number on stack with %~")
                        }
                    } else { return Err(~"stack is empty") },
                    'i' => match (copy mparams[0], copy mparams[1]) {
                        (Number(x), Number(y)) => {
                            mparams[0] = Number(x+1);
                            mparams[1] = Number(y+1);
                        },
                        (_, _) => return Err(~"first two params not numbers with %i")
                    },

                    // conditionals
                    '?' => (),
                    't' => if stack.len() > 0 {
                        match stack.pop() {
                            Number(0) => state = SeekIfElse(0),
                            Number(_) => (),
                            _         => return Err(~"non-number on stack with conditional")
                        }
                    } else { return Err(~"stack is empty") },
                    'e' => state = SeekIfEnd(0),
                    ';' => (),

                    _ => return Err(fmt!("unrecognized format option %c", cur))
                }
            },
            PushParam => {
                // params are 1-indexed
                stack.push(copy mparams[match char::to_digit(cur, 10) {
                    Some(d) => d - 1,
                    None => return Err(~"bad param number")
                }]);
            },
            SetVar => {
                if cur >= 'A' && cur <= 'Z' {
                    if stack.len() > 0 {
                        let idx = (cur as u8) - ('A' as u8);
                        vars.sta[idx] = stack.pop();
                    } else { return Err(~"stack is empty") }
                } else if cur >= 'a' && cur <= 'z' {
                    if stack.len() > 0 {
                        let idx = (cur as u8) - ('a' as u8);
                        vars.dyn[idx] = stack.pop();
                    } else { return Err(~"stack is empty") }
                } else {
                    return Err(~"bad variable name in %P");
                }
            },
            GetVar => {
                if cur >= 'A' && cur <= 'Z' {
                    let idx = (cur as u8) - ('A' as u8);
                    stack.push(copy vars.sta[idx]);
                } else if cur >= 'a' && cur <= 'z' {
                    let idx = (cur as u8) - ('a' as u8);
                    stack.push(copy vars.dyn[idx]);
                } else {
                    return Err(~"bad variable name in %g");
                }
            },
            CharConstant => {
                stack.push(Number(c as int));
                state = CharClose;
            },
            CharClose => {
                if cur != '\'' {
                    return Err(~"malformed character constant");
                }
            },
            IntConstant => {
                if cur == '}' {
                    stack.push(match int::parse_bytes(intstate, 10) {
                        Some(n) => Number(n),
                        None => return Err(~"bad int constant")
                    });
                    intstate.clear();
                    state = Nothing;
                } else {
                    intstate.push(cur as u8);
                    old_state = Nothing;
                }
            }
            SeekIfElse(level) => {
                if cur == '%' {
                    state = SeekIfElsePercent(level);
                }
                old_state = Nothing;
            }
            SeekIfElsePercent(level) => {
                if cur == ';' {
                    if level == 0 {
                        state = Nothing;
                    } else {
                        state = SeekIfElse(level-1);
                    }
                } else if cur == 'e' && level == 0 {
                    state = Nothing;
                } else if cur == '?' {
                    state = SeekIfElse(level+1);
                } else {
                    state = SeekIfElse(level);
                }
            }
            SeekIfEnd(level) => {
                if cur == '%' {
                    state = SeekIfEndPercent(level);
                }
                old_state = Nothing;
            }
            SeekIfEndPercent(level) => {
                if cur == ';' {
                    if level == 0 {
                        state = Nothing;
                    } else {
                        state = SeekIfEnd(level-1);
                    }
                } else if cur == '?' {
                    state = SeekIfEnd(level+1);
                } else {
                    state = SeekIfEnd(level);
                }
            }
        }
        if state == old_state {
            state = Nothing;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod test {
    use super::*;
    use core::result::Ok;

    #[test]
    fn test_basic_setabf() {
        let s = bytes!("\\E[48;5;%p1%dm");
        assert_eq!(expand(s, [Number(1)], &mut Variables::new()).unwrap(),
                   bytes!("\\E[48;5;1m").to_owned());
    }

    #[test]
    fn test_multiple_int_constants() {
        assert_eq!(expand(bytes!("%{1}%{2}%d%d"), [], &mut Variables::new()).unwrap(),
                   bytes!("21").to_owned());
    }

    #[test]
    fn test_op_i() {
        let mut vars = Variables::new();
        assert_eq!(expand(bytes!("%p1%d%p2%d%p3%d%i%p1%d%p2%d%p3%d"),
                          [Number(1),Number(2),Number(3)], &mut vars),
                   Ok(bytes!("123233").to_owned()));
        assert_eq!(expand(bytes!("%p1%d%p2%d%i%p1%d%p2%d"), [], &mut vars),
                   Ok(bytes!("0011").to_owned()));
    }

    #[test]
    fn test_param_stack_failure_conditions() {
        let mut varstruct = Variables::new();
        let vars = &mut varstruct;
        let caps = ["%d", "%c", "%s", "%Pa", "%l", "%!", "%~"];
        for caps.iter().advance |cap| {
            let res = expand(cap.as_bytes(), [], vars);
            assert!(res.is_err(),
                    "Op %s succeeded incorrectly with 0 stack entries", *cap);
            let p = if *cap == "%s" || *cap == "%l" { String(~"foo") } else { Number(97) };
            let res = expand((bytes!("%p1")).to_owned() + cap.as_bytes(), [p], vars);
            assert!(res.is_ok(),
                    "Op %s failed with 1 stack entry: %s", *cap, res.unwrap_err());
        }
        let caps = ["%+", "%-", "%*", "%/", "%m", "%&", "%|", "%A", "%O"];
        for caps.iter().advance |cap| {
            let res = expand(cap.as_bytes(), [], vars);
            assert!(res.is_err(),
                    "Binop %s succeeded incorrectly with 0 stack entries", *cap);
            let res = expand((bytes!("%{1}")).to_owned() + cap.as_bytes(), [], vars);
            assert!(res.is_err(),
                    "Binop %s succeeded incorrectly with 1 stack entry", *cap);
            let res = expand((bytes!("%{1}%{2}")).to_owned() + cap.as_bytes(), [], vars);
            assert!(res.is_ok(),
                    "Binop %s failed with 2 stack entries: %s", *cap, res.unwrap_err());
        }
    }

    #[test]
    fn test_push_bad_param() {
        assert!(expand(bytes!("%pa"), [], &mut Variables::new()).is_err());
    }

    #[test]
    fn test_comparison_ops() {
        let v = [('<', [1u8, 0u8, 0u8]), ('=', [0u8, 1u8, 0u8]), ('>', [0u8, 0u8, 1u8])];
        for v.iter().advance |&(op, bs)| {
            let s = fmt!("%%{1}%%{2}%%%c%%d", op);
            let res = expand(s.as_bytes(), [], &mut Variables::new());
            assert!(res.is_ok(), res.unwrap_err());
            assert_eq!(res.unwrap(), ~['0' as u8 + bs[0]]);
            let s = fmt!("%%{1}%%{1}%%%c%%d", op);
            let res = expand(s.as_bytes(), [], &mut Variables::new());
            assert!(res.is_ok(), res.unwrap_err());
            assert_eq!(res.unwrap(), ~['0' as u8 + bs[1]]);
            let s = fmt!("%%{2}%%{1}%%%c%%d", op);
            let res = expand(s.as_bytes(), [], &mut Variables::new());
            assert!(res.is_ok(), res.unwrap_err());
            assert_eq!(res.unwrap(), ~['0' as u8 + bs[2]]);
        }
    }

    #[test]
    fn test_conditionals() {
        let mut vars = Variables::new();
        let s = bytes!("\\E[%?%p1%{8}%<%t3%p1%d%e%p1%{16}%<%t9%p1%{8}%-%d%e38;5;%p1%d%;m");
        let res = expand(s, [Number(1)], &mut vars);
        assert!(res.is_ok(), res.unwrap_err());
        assert_eq!(res.unwrap(), bytes!("\\E[31m").to_owned());
        let res = expand(s, [Number(8)], &mut vars);
        assert!(res.is_ok(), res.unwrap_err());
        assert_eq!(res.unwrap(), bytes!("\\E[90m").to_owned());
        let res = expand(s, [Number(42)], &mut vars);
        assert!(res.is_ok(), res.unwrap_err());
        assert_eq!(res.unwrap(), bytes!("\\E[38;5;42m").to_owned());
    }
}
