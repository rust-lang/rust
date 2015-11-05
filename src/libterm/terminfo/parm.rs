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

pub use self::Param::*;
use self::States::*;
use self::FormatState::*;
use self::FormatOp::*;
use std::ascii::AsciiExt;
use std::mem::replace;
use std::iter::repeat;

#[derive(Copy, Clone, PartialEq)]
enum States {
    Nothing,
    Percent,
    SetVar,
    GetVar,
    PushParam,
    CharConstant,
    CharClose,
    IntConstant(isize),
    FormatPattern(Flags, FormatState),
    SeekIfElse(isize),
    SeekIfElsePercent(isize),
    SeekIfEnd(isize),
    SeekIfEndPercent(isize)
}

#[derive(Copy, Clone, PartialEq)]
enum FormatState {
    FormatStateFlags,
    FormatStateWidth,
    FormatStatePrecision
}

/// Types of parameters a capability can use
#[allow(missing_docs)]
#[derive(Clone)]
pub enum Param {
    Words(String),
    Number(isize)
}

/// Container for static and dynamic variable arrays
pub struct Variables {
    /// Static variables A-Z
    sta: [Param; 26],
    /// Dynamic variables a-z
    dyn: [Param; 26]
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

/// Expand a parameterized capability
///
/// # Arguments
/// * `cap`    - string to expand
/// * `params` - vector of params for %p1 etc
/// * `vars`   - Variables struct for %Pa etc
///
/// To be compatible with ncurses, `vars` should be the same between calls to `expand` for
/// multiple capabilities for the same terminal.
pub fn expand(cap: &[u8], params: &[Param], vars: &mut Variables)
    -> Result<Vec<u8> , String> {
    let mut state = Nothing;

    // expanded cap will only rarely be larger than the cap itself
    let mut output = Vec::with_capacity(cap.len());

    let mut stack: Vec<Param> = Vec::new();

    // Copy parameters into a local vector for mutability
    let mut mparams = [
        Number(0), Number(0), Number(0), Number(0), Number(0),
        Number(0), Number(0), Number(0), Number(0),
    ];
    for (dst, src) in mparams.iter_mut().zip(params) {
        *dst = (*src).clone();
    }

    for &c in cap {
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
                    'c' => if !stack.is_empty() {
                        match stack.pop().unwrap() {
                            // if c is 0, use 0200 (128) for ncurses compatibility
                            Number(c) => {
                                output.push(if c == 0 {
                                    128
                                } else {
                                    c as u8
                                })
                            }
                            _       => return Err("a non-char was used with %c".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    'p' => state = PushParam,
                    'P' => state = SetVar,
                    'g' => state = GetVar,
                    '\'' => state = CharConstant,
                    '{' => state = IntConstant(0),
                    'l' => if !stack.is_empty() {
                        match stack.pop().unwrap() {
                            Words(s) => stack.push(Number(s.len() as isize)),
                            _        => return Err("a non-str was used with %l".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    '+' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(x + y)),
                            _ => return Err("non-numbers on stack with +".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    '-' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(x - y)),
                            _ => return Err("non-numbers on stack with -".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    '*' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(x * y)),
                            _ => return Err("non-numbers on stack with *".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    '/' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(x / y)),
                            _ => return Err("non-numbers on stack with /".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    'm' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(x % y)),
                            _ => return Err("non-numbers on stack with %".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    '&' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(x & y)),
                            _ => return Err("non-numbers on stack with &".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    '|' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(x | y)),
                            _ => return Err("non-numbers on stack with |".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    '^' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(x ^ y)),
                            _ => return Err("non-numbers on stack with ^".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    '=' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(if x == y { 1 }
                                                                        else { 0 })),
                            _ => return Err("non-numbers on stack with =".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    '>' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(if x > y { 1 }
                                                                        else { 0 })),
                            _ => return Err("non-numbers on stack with >".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    '<' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(y), Number(x)) => stack.push(Number(if x < y { 1 }
                                                                        else { 0 })),
                            _ => return Err("non-numbers on stack with <".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    'A' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(0), Number(_)) => stack.push(Number(0)),
                            (Number(_), Number(0)) => stack.push(Number(0)),
                            (Number(_), Number(_)) => stack.push(Number(1)),
                            _ => return Err("non-numbers on stack with logical and".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    'O' => if stack.len() > 1 {
                        match (stack.pop().unwrap(), stack.pop().unwrap()) {
                            (Number(0), Number(0)) => stack.push(Number(0)),
                            (Number(_), Number(_)) => stack.push(Number(1)),
                            _ => return Err("non-numbers on stack with logical or".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    '!' => if !stack.is_empty() {
                        match stack.pop().unwrap() {
                            Number(0) => stack.push(Number(1)),
                            Number(_) => stack.push(Number(0)),
                            _ => return Err("non-number on stack with logical not".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    '~' => if !stack.is_empty() {
                        match stack.pop().unwrap() {
                            Number(x) => stack.push(Number(!x)),
                            _         => return Err("non-number on stack with %~".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    'i' => match (mparams[0].clone(), mparams[1].clone()) {
                        (Number(x), Number(y)) => {
                            mparams[0] = Number(x+1);
                            mparams[1] = Number(y+1);
                        },
                        (_, _) => return Err("first two params not numbers with %i".to_owned())
                    },

                    // printf-style support for %doxXs
                    'd'|'o'|'x'|'X'|'s' => if !stack.is_empty() {
                        let flags = Flags::new();
                        let res = format(stack.pop().unwrap(), FormatOp::from_char(cur), flags);
                        if res.is_err() { return res }
                        output.push_all(&res.unwrap())
                    } else { return Err("stack is empty".to_owned()) },
                    ':'|'#'|' '|'.'|'0'...'9' => {
                        let mut flags = Flags::new();
                        let mut fstate = FormatStateFlags;
                        match cur {
                            ':' => (),
                            '#' => flags.alternate = true,
                            ' ' => flags.space = true,
                            '.' => fstate = FormatStatePrecision,
                            '0'...'9' => {
                                flags.width = cur as usize - '0' as usize;
                                fstate = FormatStateWidth;
                            }
                            _ => unreachable!()
                        }
                        state = FormatPattern(flags, fstate);
                    }

                    // conditionals
                    '?' => (),
                    't' => if !stack.is_empty() {
                        match stack.pop().unwrap() {
                            Number(0) => state = SeekIfElse(0),
                            Number(_) => (),
                            _         => return Err("non-number on stack \
                                                    with conditional".to_owned())
                        }
                    } else { return Err("stack is empty".to_owned()) },
                    'e' => state = SeekIfEnd(0),
                    ';' => (),

                    _ => {
                        return Err(format!("unrecognized format option {:?}", cur))
                    }
                }
            },
            PushParam => {
                // params are 1-indexed
                stack.push(mparams[match cur.to_digit(10) {
                    Some(d) => d as usize - 1,
                    None => return Err("bad param number".to_owned())
                }].clone());
            },
            SetVar => {
                if cur >= 'A' && cur <= 'Z' {
                    if !stack.is_empty() {
                        let idx = (cur as u8) - b'A';
                        vars.sta[idx as usize] = stack.pop().unwrap();
                    } else { return Err("stack is empty".to_owned()) }
                } else if cur >= 'a' && cur <= 'z' {
                    if !stack.is_empty() {
                        let idx = (cur as u8) - b'a';
                        vars.dyn[idx as usize] = stack.pop().unwrap();
                    } else { return Err("stack is empty".to_owned()) }
                } else {
                    return Err("bad variable name in %P".to_owned());
                }
            },
            GetVar => {
                if cur >= 'A' && cur <= 'Z' {
                    let idx = (cur as u8) - b'A';
                    stack.push(vars.sta[idx as usize].clone());
                } else if cur >= 'a' && cur <= 'z' {
                    let idx = (cur as u8) - b'a';
                    stack.push(vars.dyn[idx as usize].clone());
                } else {
                    return Err("bad variable name in %g".to_owned());
                }
            },
            CharConstant => {
                stack.push(Number(c as isize));
                state = CharClose;
            },
            CharClose => {
                if cur != '\'' {
                    return Err("malformed character constant".to_owned());
                }
            },
            IntConstant(i) => {
                match cur {
                    '}' => {
                        stack.push(Number(i));
                        state = Nothing;
                    }
                    '0'...'9' => {
                        state = IntConstant(i*10 + (cur as isize - '0' as isize));
                        old_state = Nothing;
                    }
                    _ => return Err("bad isize constant".to_owned())
                }
            }
            FormatPattern(ref mut flags, ref mut fstate) => {
                old_state = Nothing;
                match (*fstate, cur) {
                    (_,'d')|(_,'o')|(_,'x')|(_,'X')|(_,'s') => if !stack.is_empty() {
                        let res = format(stack.pop().unwrap(), FormatOp::from_char(cur), *flags);
                        if res.is_err() { return res }
                        output.push_all(&res.unwrap());
                        // will cause state to go to Nothing
                        old_state = FormatPattern(*flags, *fstate);
                    } else { return Err("stack is empty".to_owned()) },
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
                    (FormatStateFlags,'0'...'9') => {
                        flags.width = cur as usize - '0' as usize;
                        *fstate = FormatStateWidth;
                    }
                    (FormatStateFlags,'.') => {
                        *fstate = FormatStatePrecision;
                    }
                    (FormatStateWidth,'0'...'9') => {
                        let old = flags.width;
                        flags.width = flags.width * 10 + (cur as usize - '0' as usize);
                        if flags.width < old { return Err("format width overflow".to_owned()) }
                    }
                    (FormatStateWidth,'.') => {
                        *fstate = FormatStatePrecision;
                    }
                    (FormatStatePrecision,'0'...'9') => {
                        let old = flags.precision;
                        flags.precision = flags.precision * 10 + (cur as usize - '0' as usize);
                        if flags.precision < old {
                            return Err("format precision overflow".to_owned())
                        }
                    }
                    _ => return Err("invalid format specifier".to_owned())
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

#[derive(Copy, Clone, PartialEq)]
struct Flags {
    width: usize,
    precision: usize,
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

#[derive(Copy, Clone)]
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
            _ => panic!("bad FormatOp char")
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

fn format(val: Param, op: FormatOp, flags: Flags) -> Result<Vec<u8> ,String> {
    let mut s = match val {
        Number(d) => {
            let s = match (op, flags.sign) {
                (FormatDigit, true)  => format!("{:+}", d).into_bytes(),
                (FormatDigit, false) => format!("{}", d).into_bytes(),
                (FormatOctal, _)     => format!("{:o}", d).into_bytes(),
                (FormatHex, _)       => format!("{:x}", d).into_bytes(),
                (FormatHEX, _)       => format!("{:X}", d).into_bytes(),
                (FormatString, _)    => {
                    return Err("non-number on stack with %s".to_owned())
                }
            };
            let mut s: Vec<u8> = s.into_iter().collect();
            if flags.precision > s.len() {
                let mut s_ = Vec::with_capacity(flags.precision);
                let n = flags.precision - s.len();
                s_.extend(repeat(b'0').take(n));
                s_.extend(s);
                s = s_;
            }
            assert!(!s.is_empty(), "string conversion produced empty result");
            match op {
                FormatDigit => {
                    if flags.space && !(s[0] == b'-' || s[0] == b'+' ) {
                        s.insert(0, b' ');
                    }
                }
                FormatOctal => {
                    if flags.alternate && s[0] != b'0' {
                        s.insert(0, b'0');
                    }
                }
                FormatHex => {
                    if flags.alternate {
                        let s_ = replace(&mut s, vec!(b'0', b'x'));
                        s.extend(s_);
                    }
                }
                FormatHEX => {
                    s = s.to_ascii_uppercase();
                    if flags.alternate {
                        let s_ = replace(&mut s, vec!(b'0', b'X'));
                        s.extend(s_);
                    }
                }
                FormatString => unreachable!()
            }
            s
        }
        Words(s) => {
            match op {
                FormatString => {
                    let mut s = s.as_bytes().to_vec();
                    if flags.precision > 0 && flags.precision < s.len() {
                        s.truncate(flags.precision);
                    }
                    s
                }
                _ => {
                    return Err(format!("non-string on stack with %{:?}",
                                       op.to_char()))
                }
            }
        }
    };
    if flags.width > s.len() {
        let n = flags.width - s.len();
        if flags.left {
            s.extend(repeat(b' ').take(n));
        } else {
            let mut s_ = Vec::with_capacity(flags.width);
            s_.extend(repeat(b' ').take(n));
            s_.extend(s);
            s = s_;
        }
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::{expand,Param,Words,Variables,Number};
    use std::result::Result::Ok;

    #[test]
    fn test_basic_setabf() {
        let s = b"\\E[48;5;%p1%dm";
        assert_eq!(expand(s, &[Number(1)], &mut Variables::new()).unwrap(),
                   "\\E[48;5;1m".bytes().collect::<Vec<_>>());
    }

    #[test]
    fn test_multiple_int_constants() {
        assert_eq!(expand(b"%{1}%{2}%d%d", &[], &mut Variables::new()).unwrap(),
                   "21".bytes().collect::<Vec<_>>());
    }

    #[test]
    fn test_op_i() {
        let mut vars = Variables::new();
        assert_eq!(expand(b"%p1%d%p2%d%p3%d%i%p1%d%p2%d%p3%d",
                          &[Number(1),Number(2),Number(3)], &mut vars),
                   Ok("123233".bytes().collect::<Vec<_>>()));
        assert_eq!(expand(b"%p1%d%p2%d%i%p1%d%p2%d", &[], &mut vars),
                   Ok("0011".bytes().collect::<Vec<_>>()));
    }

    #[test]
    fn test_param_stack_failure_conditions() {
        let mut varstruct = Variables::new();
        let vars = &mut varstruct;
        fn get_res(fmt: &str, cap: &str, params: &[Param], vars: &mut Variables) ->
            Result<Vec<u8>, String>
        {
            let mut u8v: Vec<_> = fmt.bytes().collect();
            u8v.extend(cap.bytes());
            expand(&u8v, params, vars)
        }

        let caps = ["%d", "%c", "%s", "%Pa", "%l", "%!", "%~"];
        for &cap in &caps {
            let res = get_res("", cap, &[], vars);
            assert!(res.is_err(),
                    "Op {} succeeded incorrectly with 0 stack entries", cap);
            let p = if cap == "%s" || cap == "%l" {
                Words(String::from("foo"))
            } else {
                Number(97)
            };
            let res = get_res("%p1", cap, &[p], vars);
            assert!(res.is_ok(),
                    "Op {} failed with 1 stack entry: {}", cap, res.err().unwrap());
        }
        let caps = ["%+", "%-", "%*", "%/", "%m", "%&", "%|", "%A", "%O"];
        for &cap in &caps {
            let res = expand(cap.as_bytes(), &[], vars);
            assert!(res.is_err(),
                    "Binop {} succeeded incorrectly with 0 stack entries", cap);
            let res = get_res("%{1}", cap, &[], vars);
            assert!(res.is_err(),
                    "Binop {} succeeded incorrectly with 1 stack entry", cap);
            let res = get_res("%{1}%{2}", cap, &[], vars);
            assert!(res.is_ok(),
                    "Binop {} failed with 2 stack entries: {:?}", cap, res.err().unwrap());
        }
    }

    #[test]
    fn test_push_bad_param() {
        assert!(expand(b"%pa", &[], &mut Variables::new()).is_err());
    }

    #[test]
    fn test_comparison_ops() {
        let v = [('<', [1, 0, 0]), ('=', [0, 1, 0]), ('>', [0, 0, 1])];
        for &(op, bs) in &v {
            let s = format!("%{{1}}%{{2}}%{}%d", op);
            let res = expand(s.as_bytes(), &[], &mut Variables::new());
            assert!(res.is_ok(), res.err().unwrap());
            assert_eq!(res.unwrap(), [b'0' + bs[0]]);
            let s = format!("%{{1}}%{{1}}%{}%d", op);
            let res = expand(s.as_bytes(), &[], &mut Variables::new());
            assert!(res.is_ok(), res.err().unwrap());
            assert_eq!(res.unwrap(), [b'0' + bs[1]]);
            let s = format!("%{{2}}%{{1}}%{}%d", op);
            let res = expand(s.as_bytes(), &[], &mut Variables::new());
            assert!(res.is_ok(), res.err().unwrap());
            assert_eq!(res.unwrap(), [b'0' + bs[2]]);
        }
    }

    #[test]
    fn test_conditionals() {
        let mut vars = Variables::new();
        let s = b"\\E[%?%p1%{8}%<%t3%p1%d%e%p1%{16}%<%t9%p1%{8}%-%d%e38;5;%p1%d%;m";
        let res = expand(s, &[Number(1)], &mut vars);
        assert!(res.is_ok(), res.err().unwrap());
        assert_eq!(res.unwrap(),
                   "\\E[31m".bytes().collect::<Vec<_>>());
        let res = expand(s, &[Number(8)], &mut vars);
        assert!(res.is_ok(), res.err().unwrap());
        assert_eq!(res.unwrap(),
                   "\\E[90m".bytes().collect::<Vec<_>>());
        let res = expand(s, &[Number(42)], &mut vars);
        assert!(res.is_ok(), res.err().unwrap());
        assert_eq!(res.unwrap(),
                   "\\E[38;5;42m".bytes().collect::<Vec<_>>());
    }

    #[test]
    fn test_format() {
        let mut varstruct = Variables::new();
        let vars = &mut varstruct;
        assert_eq!(expand(b"%p1%s%p2%2s%p3%2s%p4%.2s",
                          &[Words(String::from("foo")),
                            Words(String::from("foo")),
                            Words(String::from("f")),
                            Words(String::from("foo"))], vars),
                   Ok("foofoo ffo".bytes().collect::<Vec<_>>()));
        assert_eq!(expand(b"%p1%:-4.2s", &[Words("foo".to_owned())], vars),
                   Ok("fo  ".bytes().collect::<Vec<_>>()));

        assert_eq!(expand(b"%p1%d%p1%.3d%p1%5d%p1%:+d", &[Number(1)], vars),
                   Ok("1001    1+1".bytes().collect::<Vec<_>>()));
        assert_eq!(expand(b"%p1%o%p1%#o%p2%6.4x%p2%#6.4X", &[Number(15), Number(27)], vars),
                   Ok("17017  001b0X001B".bytes().collect::<Vec<_>>()));
    }
}
