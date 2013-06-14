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
    IfCond,
    IfBody
}

/// Types of parameters a capability can use
pub enum Param {
    String(~str),
    Char(char),
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
    let mut i = 0;

    // expanded cap will only rarely be larger than the cap itself
    let mut output = vec::with_capacity(cap.len());

    let mut cur;

    let mut stack: ~[Param] = ~[];

    let mut intstate = ~[];

    // Copy parameters into a local vector for mutability
    let mut mparams = [Number(0), ..9];
    for mparams.mut_iter().zip(params.iter()).advance |(dst, &src)| {
        *dst = src;
    }

    while i < cap.len() {
        cur = cap[i] as char;
        let mut old_state = state;
        match state {
            Nothing => {
                if cur == '%' {
                    state = Percent;
                } else {
                    output.push(cap[i]);
                }
            },
            Percent => {
                match cur {
                    '%' => { output.push(cap[i]); state = Nothing },
                    'c' => match stack.pop() {
                        Char(c) => output.push(c as u8),
                        _       => return Err(~"a non-char was used with %c")
                    },
                    's' => match stack.pop() {
                        String(s) => output.push_all(s.as_bytes()),
                        _         => return Err(~"a non-str was used with %s")
                    },
                    'd' => match stack.pop() {
                        Number(x) => {
                            let s = x.to_str();
                            output.push_all(s.as_bytes())
                        }
                        _         => return Err(~"a non-number was used with %d")
                    },
                    'p' => state = PushParam,
                    'P' => state = SetVar,
                    'g' => state = GetVar,
                    '\'' => state = CharConstant,
                    '{' => state = IntConstant,
                    'l' => match stack.pop() {
                        String(s) => stack.push(Number(s.len() as int)),
                        _         => return Err(~"a non-str was used with %l")
                    },
                    '+' => match (stack.pop(), stack.pop()) {
                        (Number(x), Number(y)) => stack.push(Number(x + y)),
                        (_, _) => return Err(~"non-numbers on stack with +")
                    },
                    '-' => match (stack.pop(), stack.pop()) {
                        (Number(x), Number(y)) => stack.push(Number(x - y)),
                        (_, _) => return Err(~"non-numbers on stack with -")
                    },
                    '*' => match (stack.pop(), stack.pop()) {
                        (Number(x), Number(y)) => stack.push(Number(x * y)),
                        (_, _) => return Err(~"non-numbers on stack with *")
                    },
                    '/' => match (stack.pop(), stack.pop()) {
                        (Number(x), Number(y)) => stack.push(Number(x / y)),
                        (_, _) => return Err(~"non-numbers on stack with /")
                    },
                    'm' => match (stack.pop(), stack.pop()) {
                        (Number(x), Number(y)) => stack.push(Number(x % y)),
                        (_, _) => return Err(~"non-numbers on stack with %")
                    },
                    '&' => match (stack.pop(), stack.pop()) {
                        (Number(x), Number(y)) => stack.push(Number(x & y)),
                        (_, _) => return Err(~"non-numbers on stack with &")
                    },
                    '|' => match (stack.pop(), stack.pop()) {
                        (Number(x), Number(y)) => stack.push(Number(x | y)),
                        (_, _) => return Err(~"non-numbers on stack with |")
                    },
                    'A' => match (stack.pop(), stack.pop()) {
                        (Number(0), Number(_)) => stack.push(Number(0)),
                        (Number(_), Number(0)) => stack.push(Number(0)),
                        (Number(_), Number(_)) => stack.push(Number(1)),
                        _ => return Err(~"non-numbers on stack with logical and")
                    },
                    'O' => match (stack.pop(), stack.pop()) {
                        (Number(0), Number(0)) => stack.push(Number(0)),
                        (Number(_), Number(_)) => stack.push(Number(1)),
                        _ => return Err(~"non-numbers on stack with logical or")
                    },
                    '!' => match stack.pop() {
                        Number(0) => stack.push(Number(1)),
                        Number(_) => stack.push(Number(0)),
                        _ => return Err(~"non-number on stack with logical not")
                    },
                    '~' => match stack.pop() {
                        Number(x) => stack.push(Number(!x)),
                        _         => return Err(~"non-number on stack with %~")
                    },
                    'i' => match (copy mparams[0], copy mparams[1]) {
                        (Number(ref mut x), Number(ref mut y)) => {
                            *x += 1;
                            *y += 1;
                        },
                        (_, _) => return Err(~"first two params not numbers with %i")
                    },
                    '?' => state = return Err(fmt!("if expressions unimplemented (%?)", cap)),
                    _ => return Err(fmt!("unrecognized format option %c", cur))
                }
            },
            PushParam => {
                // params are 1-indexed
                stack.push(copy mparams[char::to_digit(cur, 10).expect("bad param number") - 1]);
            },
            SetVar => {
                if cur >= 'A' && cur <= 'Z' {
                    let idx = (cur as u8) - ('A' as u8);
                    vars.sta[idx] = stack.pop();
                } else if cur >= 'a' && cur <= 'z' {
                    let idx = (cur as u8) - ('a' as u8);
                    vars.dyn[idx] = stack.pop();
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
                stack.push(Char(cur));
                state = CharClose;
            },
            CharClose => {
                if cur != '\'' {
                    return Err(~"malformed character constant");
                }
            },
            IntConstant => {
                if cur == '}' {
                    stack.push(Number(int::parse_bytes(intstate, 10).expect("bad int constant")));
                    state = Nothing;
                }
                intstate.push(cur as u8);
                old_state = Nothing;
            }
            _ => return Err(~"unimplemented state")
        }
        if state == old_state {
            state = Nothing;
        }
        i += 1;
    }
    Ok(output)
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_basic_setabf() {
        let s = bytes!("\\E[48;5;%p1%dm");
        assert_eq!(expand(s, [Number(1)], &mut Variables::new()).unwrap(), bytes!("\\E[48;5;1m").to_owned());
    }
}
