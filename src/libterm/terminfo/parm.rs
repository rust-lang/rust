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

use std::{char, vec};
use std::mem::replace;

#[deriving(Eq)]
enum States {
    Nothing,
    Percent,
    SetVar,
    GetVar,
    PushParam,
    CharConstant,
    CharClose,
    IntConstant(int),
    FormatPattern(Flags, FormatState),
    SeekIfElse(int),
    SeekIfElsePercent(int),
    SeekIfEnd(int),
    SeekIfEndPercent(int)
}

#[deriving(Eq)]
enum FormatState {
    FormatStateFlags,
    FormatStateWidth,
    FormatStatePrecision
}

/// Types of parameters a capability can use
#[deriving(Clone)]
#[allow(missing_doc)]
pub enum Param {
    String(~str),
    Number(int)
}

/// Container for static and dynamic variable arrays
pub struct Variables {
    /// Static variables A-Z
    priv sta: [Param, ..26],
    /// Dynamic variables a-z
    priv dyn: [Param, ..26]
}

impl Variables {
    /// Return a new zero-initialized Variables
    pub fn new() -> Variables {
        Variables {
            sta: [
                Number(0), Number(0), Number(0), Number(0), Number(0),
                Number(0), Number(0), Number(0), Number(0), Number(0),
                Number(0), Number(0), Number(0), Number(0), Number(0),
                Number(0), Number(0), Number(0), Number(0), Number(0),
                Number(0), Number(0), Number(0), Number(0), Number(0),
                Number(0),
            ],
            dyn: [
                Number(0), Number(0), Number(0), Number(0), Number(0),
                Number(0), Number(0), Number(0), Number(0), Number(0),
                Number(0), Number(0), Number(0), Number(0), Number(0),
                Number(0), Number(0), Number(0), Number(0), Number(0),
                Number(0), Number(0), Number(0), Number(0), Number(0),
                Number(0),
            ],
        }
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

    // Copy parameters into a local vector for mutability
    let mut mparams = [
        Number(0), Number(0), Number(0), Number(0), Number(0),
        Number(0), Number(0), Number(0), Number(0),
    ];
    for (dst, src) in mparams.mut_iter().zip(params.iter()) {
        *dst = (*src).clone();
    }

    for c in cap.iter().map(|&x| x) {
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
                        match stack.pop().unwrap() {
                            // if c is 0, use 0200 (128) for ncurses compatibility
                            Number(c) => output.push(if c == 0 { 128 } else { c } as u8),
                            _       => return Err(~"a non-char was used with %c")
                        }
                    } else { return Err(~"stack is empty") },
                    'p' => state = PushParam,
                    'P' => state = SetVar,
                    'g' => state = GetVar,
                    '\'' => state = CharConstant,
                    '{' => state = IntConstant(0),
                    'l' => if stack.len() > 0 {
                        match stack.pop().unwrap() {
                            String(s) => stack.push(Number(s.len() as int)),
                            _         => return Err(~"a non-str was used with %l")
                        }
                    } else { return Err(~"stack is empty") },
                    '+' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(x + y)),
                            _ => return Err(~"non-numbers on stack with +")
                        }
                    } else { return Err(~"stack is empty") },
                    '-' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(x - y)),
                            _ => return Err(~"non-numbers on stack with -")
                        }
                    } else { return Err(~"stack is empty") },
                    '*' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(x * y)),
                            _ => return Err(~"non-numbers on stack with *")
                        }
                    } else { return Err(~"stack is empty") },
                    '/' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(x / y)),
                            _ => return Err(~"non-numbers on stack with /")
                        }
                    } else { return Err(~"stack is empty") },
                    'm' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(x % y)),
                            _ => return Err(~"non-numbers on stack with %")
                        }
                    } else { return Err(~"stack is empty") },
                    '&' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(x & y)),
                            _ => return Err(~"non-numbers on stack with &")
                        }
                    } else { return Err(~"stack is empty") },
                    '|' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(x | y)),
                            _ => return Err(~"non-numbers on stack with |")
                        }
                    } else { return Err(~"stack is empty") },
                    '^' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(x ^ y)),
                            _ => return Err(~"non-numbers on stack with ^")
                        }
                    } else { return Err(~"stack is empty") },
                    '=' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(if x == y { 1 }
                                                                        else { 0 })),
                            _ => return Err(~"non-numbers on stack with =")
                        }
                    } else { return Err(~"stack is empty") },
                    '>' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(if x > y { 1 }
                                                                        else { 0 })),
                            _ => return Err(~"non-numbers on stack with >")
                        }
                    } else { return Err(~"stack is empty") },
                    '<' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(if x < y { 1 }
                                                                        else { 0 })),
                            _ => return Err(~"non-numbers on stack with <")
                        }
                    } else { return Err(~"stack is empty") },
                    'A' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(0), Number(_)) => stack.push(Number(0)),
                            (Number(_), Number(0)) => stack.push(Number(0)),
                            (Number(_), Number(_)) => stack.push(Number(1)),
                            _ => return Err(~"non-numbers on stack with logical and")
                        }
                    } else { return Err(~"stack is empty") },
                    'O' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(0), Number(0)) => stack.push(Number(0)),
                            (Number(_), Number(_)) => stack.push(Number(1)),
                            _ => return Err(~"non-numbers on stack with logical or")
                        }
                    } else { return Err(~"stack is empty") },
                    '!' => if stack.len() > 0 {
                        match stack.pop().unwrap() {
                            Number(0) => stack.push(Number(1)),
                            Number(_) => stack.push(Number(0)),
                            _ => return Err(~"non-number on stack with logical not")
                        }
                    } else { return Err(~"stack is empty") },
                    '~' => if stack.len() > 0 {
                        match stack.pop().unwrap() {
                            Number(x) => stack.push(Number(!x)),
                            _         => return Err(~"non-number on stack with %~")
                        }
                    } else { return Err(~"stack is empty") },
                    'i' => match (mparams[0].clone(), mparams[1].clone()) {
                        (Number(x), Number(y)) => {
                            mparams[0] = Number(x+1);
                            mparams[1] = Number(y+1);
                        },
                        (_, _) => return Err(~"first two params not numbers with %i")
                    },

                    // printf-style support for %doxXs
                    'd'|'o'|'x'|'X'|'s' => if stack.len() > 0 {
                        let flags = Flags::new();
                        let res = format(stack.pop().unwrap(), FormatOp::from_char(cur), flags);
                        if res.is_err() { return res }
                        output.push_all(res.unwrap())
                    } else { return Err(~"stack is empty") },
                    ':'|'#'|' '|'.'|'0'..'9' => {
                        let mut flags = Flags::new();
                        let mut fstate = FormatStateFlags;
                        match cur {
                            ':' => (),
                            '#' => flags.alternate = true,
                            ' ' => flags.space = true,
                            '.' => fstate = FormatStatePrecision,
                            '0'..'9' => {
                                flags.width = cur as uint - '0' as uint;
                                fstate = FormatStateWidth;
                            }
                            _ => unreachable!()
                        }
                        state = FormatPattern(flags, fstate);
                    }

                    // conditionals
                    '?' => (),
                    't' => if stack.len() > 0 {
                        match stack.pop().unwrap() {
                            Number(0) => state = SeekIfElse(0),
                            Number(_) => (),
                            _         => return Err(~"non-number on stack with conditional")
                        }
                    } else { return Err(~"stack is empty") },
                    'e' => state = SeekIfEnd(0),
                    ';' => (),

                    _ => return Err(format!("unrecognized format option {}", cur))
                }
            },
            PushParam => {
                // params are 1-indexed
                stack.push(mparams[match char::to_digit(cur, 10) {
                    Some(d) => d - 1,
                    None => return Err(~"bad param number")
                }].clone());
            },
            SetVar => {
                if cur >= 'A' && cur <= 'Z' {
                    if stack.len() > 0 {
                        let idx = (cur as u8) - ('A' as u8);
                        vars.sta[idx] = stack.pop().unwrap();
                    } else { return Err(~"stack is empty") }
                } else if cur >= 'a' && cur <= 'z' {
                    if stack.len() > 0 {
                        let idx = (cur as u8) - ('a' as u8);
                        vars.dyn[idx] = stack.pop().unwrap();
                    } else { return Err(~"stack is empty") }
                } else {
                    return Err(~"bad variable name in %P");
                }
            },
            GetVar => {
                if cur >= 'A' && cur <= 'Z' {
                    let idx = (cur as u8) - ('A' as u8);
                    stack.push(vars.sta[idx].clone());
                } else if cur >= 'a' && cur <= 'z' {
                    let idx = (cur as u8) - ('a' as u8);
                    stack.push(vars.dyn[idx].clone());
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
            IntConstant(i) => {
                match cur {
                    '}' => {
                        stack.push(Number(i));
                        state = Nothing;
                    }
                    '0'..'9' => {
                        state = IntConstant(i*10 + (cur as int - '0' as int));
                        old_state = Nothing;
                    }
                    _ => return Err(~"bad int constant")
                }
            }
            FormatPattern(ref mut flags, ref mut fstate) => {
                old_state = Nothing;
                match (*fstate, cur) {
                    (_,'d')|(_,'o')|(_,'x')|(_,'X')|(_,'s') => if stack.len() > 0 {
                        let res = format(stack.pop().unwrap(), FormatOp::from_char(cur), *flags);
                        if res.is_err() { return res }
                        output.push_all(res.unwrap());
                        old_state = state; // will cause state to go to Nothing
                    } else { return Err(~"stack is empty") },
                    (FormatStateFlags,'#') => {
                        flags.alternate = true;
                    }
                    (FormatStateFlags,'-') => {
                        flags.left = true;
                    }
                    (FormatStateFlags,'+') => {
                        flags.sign = true;
                    }
                    (FormatStateFlags,' ') => {
                        flags.space = true;
                    }
                    (FormatStateFlags,'0'..'9') => {
                        flags.width = cur as uint - '0' as uint;
                        *fstate = FormatStateWidth;
                    }
                    (FormatStateFlags,'.') => {
                        *fstate = FormatStatePrecision;
                    }
                    (FormatStateWidth,'0'..'9') => {
                        let old = flags.width;
                        flags.width = flags.width * 10 + (cur as uint - '0' as uint);
                        if flags.width < old { return Err(~"format width overflow") }
                    }
                    (FormatStateWidth,'.') => {
                        *fstate = FormatStatePrecision;
                    }
                    (FormatStatePrecision,'0'..'9') => {
                        let old = flags.precision;
                        flags.precision = flags.precision * 10 + (cur as uint - '0' as uint);
                        if flags.precision < old { return Err(~"format precision overflow") }
                    }
                    _ => return Err(~"invalid format specifier")
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

#[deriving(Eq)]
struct Flags {
    width: uint,
    precision: uint,
    alternate: bool,
    left: bool,
    sign: bool,
    space: bool
}

impl Flags {
    fn new() -> Flags {
        Flags{ width: 0, precision: 0, alternate: false,
               left: false, sign: false, space: false }
    }
}

enum FormatOp {
    FormatDigit,
    FormatOctal,
    FormatHex,
    FormatHEX,
    FormatString
}

impl FormatOp {
    fn from_char(c: char) -> FormatOp {
        match c {
            'd' => FormatDigit,
            'o' => FormatOctal,
            'x' => FormatHex,
            'X' => FormatHEX,
            's' => FormatString,
            _ => fail!("bad FormatOp char")
        }
    }
    fn to_char(self) -> char {
        match self {
            FormatDigit => 'd',
            FormatOctal => 'o',
            FormatHex => 'x',
            FormatHEX => 'X',
            FormatString => 's'
        }
    }
}

fn format(val: Param, op: FormatOp, flags: Flags) -> Result<~[u8],~str> {
    let mut s = match val {
        Number(d) => {
            let mut s = match (op, flags.sign) {
                (FormatDigit, true)  => format!("{:+d}", d).into_bytes(),
                (FormatDigit, false) => format!("{:d}", d).into_bytes(),
                (FormatOctal, _)     => format!("{:o}", d).into_bytes(),
                (FormatHex, _)       => format!("{:x}", d).into_bytes(),
                (FormatHEX, _)       => format!("{:X}", d).into_bytes(),
                (FormatString, _)    => return Err(~"non-number on stack with %s"),
            };
            if flags.precision > s.len() {
                let mut s_ = vec::with_capacity(flags.precision);
                let n = flags.precision - s.len();
                s_.grow(n, &('0' as u8));
                s_.push_all_move(s);
                s = s_;
            }
            assert!(!s.is_empty(), "string conversion produced empty result");
            match op {
                FormatDigit => {
                    if flags.space && !(s[0] == '-' as u8 || s[0] == '+' as u8) {
                        s.unshift(' ' as u8);
                    }
                }
                FormatOctal => {
                    if flags.alternate && s[0] != '0' as u8 {
                        s.unshift('0' as u8);
                    }
                }
                FormatHex => {
                    if flags.alternate {
                        let s_ = replace(&mut s, ~['0' as u8, 'x' as u8]);
                        s.push_all_move(s_);
                    }
                }
                FormatHEX => {
                    s = s.into_ascii().to_upper().into_bytes();
                    if flags.alternate {
                        let s_ = replace(&mut s, ~['0' as u8, 'X' as u8]);
                        s.push_all_move(s_);
                    }
                }
                FormatString => unreachable!()
            }
            s
        }
        String(s) => {
            match op {
                FormatString => {
                    let mut s = s.as_bytes().to_owned();
                    if flags.precision > 0 && flags.precision < s.len() {
                        s.truncate(flags.precision);
                    }
                    s
                }
                _ => {
                    return Err(format!("non-string on stack with %{}", op.to_char()))
                }
            }
        }
    };
    if flags.width > s.len() {
        let n = flags.width - s.len();
        if flags.left {
            s.grow(n, &(' ' as u8));
        } else {
            let mut s_ = vec::with_capacity(flags.width);
            s_.grow(n, &(' ' as u8));
            s_.push_all_move(s);
            s = s_;
        }
    }
    Ok(s)
}

#[cfg(test)]
mod test {
    use super::{expand,String,Variables,Number};
    use std::result::Ok;

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
        for cap in caps.iter() {
            let res = expand(cap.as_bytes(), [], vars);
            assert!(res.is_err(),
                    "Op {} succeeded incorrectly with 0 stack entries", *cap);
            let p = if *cap == "%s" || *cap == "%l" { String(~"foo") } else { Number(97) };
            let res = expand((bytes!("%p1")).to_owned() + cap.as_bytes(), [p], vars);
            assert!(res.is_ok(),
                    "Op {} failed with 1 stack entry: {}", *cap, res.unwrap_err());
        }
        let caps = ["%+", "%-", "%*", "%/", "%m", "%&", "%|", "%A", "%O"];
        for cap in caps.iter() {
            let res = expand(cap.as_bytes(), [], vars);
            assert!(res.is_err(),
                    "Binop {} succeeded incorrectly with 0 stack entries", *cap);
            let res = expand((bytes!("%{1}")).to_owned() + cap.as_bytes(), [], vars);
            assert!(res.is_err(),
                    "Binop {} succeeded incorrectly with 1 stack entry", *cap);
            let res = expand((bytes!("%{1}%{2}")).to_owned() + cap.as_bytes(), [], vars);
            assert!(res.is_ok(),
                    "Binop {} failed with 2 stack entries: {}", *cap, res.unwrap_err());
        }
    }

    #[test]
    fn test_push_bad_param() {
        assert!(expand(bytes!("%pa"), [], &mut Variables::new()).is_err());
    }

    #[test]
    fn test_comparison_ops() {
        let v = [('<', [1u8, 0u8, 0u8]), ('=', [0u8, 1u8, 0u8]), ('>', [0u8, 0u8, 1u8])];
        for &(op, bs) in v.iter() {
            let s = format!("%\\{1\\}%\\{2\\}%{}%d", op);
            let res = expand(s.as_bytes(), [], &mut Variables::new());
            assert!(res.is_ok(), res.unwrap_err());
            assert_eq!(res.unwrap(), ~['0' as u8 + bs[0]]);
            let s = format!("%\\{1\\}%\\{1\\}%{}%d", op);
            let res = expand(s.as_bytes(), [], &mut Variables::new());
            assert!(res.is_ok(), res.unwrap_err());
            assert_eq!(res.unwrap(), ~['0' as u8 + bs[1]]);
            let s = format!("%\\{2\\}%\\{1\\}%{}%d", op);
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

    #[test]
    fn test_format() {
        let mut varstruct = Variables::new();
        let vars = &mut varstruct;
        assert_eq!(expand(bytes!("%p1%s%p2%2s%p3%2s%p4%.2s"),
                          [String(~"foo"), String(~"foo"), String(~"f"), String(~"foo")], vars),
                   Ok(bytes!("foofoo ffo").to_owned()));
        assert_eq!(expand(bytes!("%p1%:-4.2s"), [String(~"foo")], vars),
                   Ok(bytes!("fo  ").to_owned()));

        assert_eq!(expand(bytes!("%p1%d%p1%.3d%p1%5d%p1%:+d"), [Number(1)], vars),
                   Ok(bytes!("1001    1+1").to_owned()));
        assert_eq!(expand(bytes!("%p1%o%p1%#o%p2%6.4x%p2%#6.4X"), [Number(15), Number(27)], vars),
                   Ok(bytes!("17017  001b0X001B").to_owned()));
    }
}
