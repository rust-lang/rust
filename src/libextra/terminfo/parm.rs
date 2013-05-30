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

pub enum Param {
    String(~str),
    Char(char),
    Number(int)
}

pub fn expand(cap: &[u8], params: &mut [Param], sta: &mut [Param], dyn: &mut [Param]) -> ~[u8] {
    assert!(cap.len() != 0, "expanding an empty capability makes no sense");
    assert!(params.len() <= 9, "only 9 parameters are supported by capability strings");

    assert!(sta.len() <= 26, "only 26 static vars are able to be used by capability strings");
    assert!(dyn.len() <= 26, "only 26 dynamic vars are able to be used by capability strings");

    let mut state = Nothing;
    let mut i = 0;

    // expanded cap will only rarely be smaller than the cap itself
    let mut output = vec::with_capacity(cap.len());

    let mut cur;

    let mut stack: ~[Param] = ~[];

    let mut intstate = ~[];

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
                        _       => fail!("a non-char was used with %c")
                    },
                    's' => match stack.pop() {
                        String(s) => output.push_all(s.to_bytes()),
                        _         => fail!("a non-str was used with %s")
                    },
                    'd' => match stack.pop() {
                        Number(x) => output.push_all(x.to_str().to_bytes()),
                        _         => fail!("a non-number was used with %d")
                    },
                    'p' => state = PushParam,
                    'P' => state = SetVar,
                    'g' => state = GetVar,
                    '\'' => state = CharConstant,
                    '{' => state = IntConstant,
                    'l' => match stack.pop() {
                        String(s) => stack.push(Number(s.len() as int)),
                        _         => fail!("a non-str was used with %l")
                    },
                    '+' => match (stack.pop(), stack.pop()) {
                        (Number(x), Number(y)) => stack.push(Number(x + y)),
                        (_, _) => fail!("non-numbers on stack with +")
                    },
                    '-' => match (stack.pop(), stack.pop()) {
                        (Number(x), Number(y)) => stack.push(Number(x - y)),
                        (_, _) => fail!("non-numbers on stack with -")
                    },
                    '*' => match (stack.pop(), stack.pop()) {
                        (Number(x), Number(y)) => stack.push(Number(x * y)),
                        (_, _) => fail!("non-numbers on stack with *")
                    },
                    '/' => match (stack.pop(), stack.pop()) {
                        (Number(x), Number(y)) => stack.push(Number(x / y)),
                        (_, _) => fail!("non-numbers on stack with /")
                    },
                    'm' => match (stack.pop(), stack.pop()) {
                        (Number(x), Number(y)) => stack.push(Number(x % y)),
                        (_, _) => fail!("non-numbers on stack with %")
                    },
                    '&' => match (stack.pop(), stack.pop()) {
                        (Number(x), Number(y)) => stack.push(Number(x & y)),
                        (_, _) => fail!("non-numbers on stack with &")
                    },
                    '|' => match (stack.pop(), stack.pop()) {
                        (Number(x), Number(y)) => stack.push(Number(x | y)),
                        (_, _) => fail!("non-numbers on stack with |")
                    },
                    'A' => fail!("logical operations unimplemented"),
                    'O' => fail!("logical operations unimplemented"),
                    '!' => fail!("logical operations unimplemented"),
                    '~' => match stack.pop() {
                        Number(x) => stack.push(Number(!x)),
                        _         => fail!("non-number on stack with %~")
                    },
                    'i' => match (copy params[0], copy params[1]) {
                        (Number(x), Number(y)) => {
                            params[0] = Number(x + 1);
                            params[1] = Number(y + 1);
                        },
                        (_, _) => fail!("first two params not numbers with %i")
                    },
                    '?' => state = fail!("if expressions unimplemented"),
                    _ => fail!("unrecognized format option %c", cur)
                }
            },
            PushParam => {
                // params are 1-indexed
                stack.push(copy params[char::to_digit(cur, 10).expect("bad param number") - 1]);
            },
            SetVar => {
                if cur >= 'A' && cur <= 'Z' {
                    let idx = (cur as u8) - ('A' as u8);
                    sta[idx] = stack.pop();
                } else if cur >= 'a' && cur <= 'z' {
                    let idx = (cur as u8) - ('a' as u8);
                    dyn[idx] = stack.pop();
                } else {
                    fail!("bad variable name in %P");
                }
            },
            GetVar => {
                if cur >= 'A' && cur <= 'Z' {
                    let idx = (cur as u8) - ('A' as u8);
                    stack.push(copy sta[idx]);
                } else if cur >= 'a' && cur <= 'z' {
                    let idx = (cur as u8) - ('a' as u8);
                    stack.push(copy dyn[idx]);
                } else {
                    fail!("bad variable name in %g");
                }
            },
            CharConstant => {
                stack.push(Char(cur));
                state = CharClose;
            },
            CharClose => {
                assert!(cur == '\'', "malformed character constant");
            },
            IntConstant => {
                if cur == '}' {
                    stack.push(Number(int::parse_bytes(intstate, 10).expect("bad int constant")));
                    state = Nothing;
                }
                intstate.push(cur as u8);
                old_state = Nothing;
            }
            _ => fail!("unimplemented state")
        }
        if state == old_state {
            state = Nothing;
        }
        i += 1;
    }
    output
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_basic_setabf() {
        let s = bytes!("\\E[48;5;%p1%dm");
        assert_eq!(expand(s, [Number(1)], [], []), bytes!("\\E[48;5;1m").to_owned());
    }
}
