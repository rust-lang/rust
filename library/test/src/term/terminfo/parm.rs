//! Parameterized string expansion

use self::Param::*;
use self::States::*;

use std::iter::repeat;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, PartialEq)]
enum States {
    Nothing,
    Percent,
    SetVar,
    GetVar,
    PushParam,
    CharConstant,
    CharClose,
    IntConstant(i32),
    FormatPattern(Flags, FormatState),
    SeekIfElse(usize),
    SeekIfElsePercent(usize),
    SeekIfEnd(usize),
    SeekIfEndPercent(usize),
}

#[derive(Copy, PartialEq, Clone)]
enum FormatState {
    Flags,
    Width,
    Precision,
}

/// Types of parameters a capability can use
#[allow(missing_docs)]
#[derive(Clone)]
pub(crate) enum Param {
    Number(i32),
}

/// Container for static and dynamic variable arrays
pub(crate) struct Variables {
    /// Static variables A-Z
    sta_va: [Param; 26],
    /// Dynamic variables a-z
    dyn_va: [Param; 26],
}

impl Variables {
    /// Returns a new zero-initialized Variables
    pub(crate) fn new() -> Variables {
        Variables {
            sta_va: [
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
            ],
            dyn_va: [
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
                Number(0),
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
pub(crate) fn expand(
    cap: &[u8],
    params: &[Param],
    vars: &mut Variables,
) -> Result<Vec<u8>, String> {
    let mut state = Nothing;

    // expanded cap will only rarely be larger than the cap itself
    let mut output = Vec::with_capacity(cap.len());

    let mut stack: Vec<Param> = Vec::new();

    // Copy parameters into a local vector for mutability
    let mut mparams = [
        Number(0),
        Number(0),
        Number(0),
        Number(0),
        Number(0),
        Number(0),
        Number(0),
        Number(0),
        Number(0),
    ];
    for (dst, src) in mparams.iter_mut().zip(params.iter()) {
        *dst = (*src).clone();
    }

    for &c in cap.iter() {
        let cur = c as char;
        let mut old_state = state;
        match state {
            Nothing => {
                if cur == '%' {
                    state = Percent;
                } else {
                    output.push(c);
                }
            }
            Percent => {
                match cur {
                    '%' => {
                        output.push(c);
                        state = Nothing
                    }
                    'c' => {
                        match stack.pop() {
                            // if c is 0, use 0200 (128) for ncurses compatibility
                            Some(Number(0)) => output.push(128u8),
                            // Don't check bounds. ncurses just casts and truncates.
                            Some(Number(c)) => output.push(c as u8),
                            None => return Err("stack is empty".to_string()),
                        }
                    }
                    'p' => state = PushParam,
                    'P' => state = SetVar,
                    'g' => state = GetVar,
                    '\'' => state = CharConstant,
                    '{' => state = IntConstant(0),
                    'l' => match stack.pop() {
                        Some(_) => return Err("a non-str was used with %l".to_string()),
                        None => return Err("stack is empty".to_string()),
                    },
                    '+' | '-' | '/' | '*' | '^' | '&' | '|' | 'm' => {
                        match (stack.pop(), stack.pop()) {
                            (Some(Number(y)), Some(Number(x))) => stack.push(Number(match cur {
                                '+' => x + y,
                                '-' => x - y,
                                '*' => x * y,
                                '/' => x / y,
                                '|' => x | y,
                                '&' => x & y,
                                '^' => x ^ y,
                                'm' => x % y,
                                _ => unreachable!("All cases handled"),
                            })),
                            _ => return Err("stack is empty".to_string()),
                        }
                    }
                    '=' | '>' | '<' | 'A' | 'O' => match (stack.pop(), stack.pop()) {
                        (Some(Number(y)), Some(Number(x))) => stack.push(Number(
                            if match cur {
                                '=' => x == y,
                                '<' => x < y,
                                '>' => x > y,
                                'A' => x > 0 && y > 0,
                                'O' => x > 0 || y > 0,
                                _ => unreachable!(),
                            } {
                                1
                            } else {
                                0
                            },
                        )),
                        _ => return Err("stack is empty".to_string()),
                    },
                    '!' | '~' => match stack.pop() {
                        Some(Number(x)) => stack.push(Number(match cur {
                            '!' if x > 0 => 0,
                            '!' => 1,
                            '~' => !x,
                            _ => unreachable!(),
                        })),
                        None => return Err("stack is empty".to_string()),
                    },
                    'i' => match (&mparams[0], &mparams[1]) {
                        (&Number(x), &Number(y)) => {
                            mparams[0] = Number(x + 1);
                            mparams[1] = Number(y + 1);
                        }
                    },

                    // printf-style support for %doxXs
                    'd' | 'o' | 'x' | 'X' | 's' => {
                        if let Some(arg) = stack.pop() {
                            let flags = Flags::new();
                            let res = format(arg, FormatOp::from_char(cur), flags)?;
                            output.extend(res.iter().cloned());
                        } else {
                            return Err("stack is empty".to_string());
                        }
                    }
                    ':' | '#' | ' ' | '.' | '0'..='9' => {
                        let mut flags = Flags::new();
                        let mut fstate = FormatState::Flags;
                        match cur {
                            ':' => (),
                            '#' => flags.alternate = true,
                            ' ' => flags.space = true,
                            '.' => fstate = FormatState::Precision,
                            '0'..='9' => {
                                flags.width = cur as usize - '0' as usize;
                                fstate = FormatState::Width;
                            }
                            _ => unreachable!(),
                        }
                        state = FormatPattern(flags, fstate);
                    }

                    // conditionals
                    '?' => (),
                    't' => match stack.pop() {
                        Some(Number(0)) => state = SeekIfElse(0),
                        Some(Number(_)) => (),
                        None => return Err("stack is empty".to_string()),
                    },
                    'e' => state = SeekIfEnd(0),
                    ';' => (),
                    _ => return Err(format!("unrecognized format option {}", cur)),
                }
            }
            PushParam => {
                // params are 1-indexed
                stack.push(
                    mparams[match cur.to_digit(10) {
                        Some(d) => d as usize - 1,
                        None => return Err("bad param number".to_string()),
                    }]
                    .clone(),
                );
            }
            SetVar => {
                if cur >= 'A' && cur <= 'Z' {
                    if let Some(arg) = stack.pop() {
                        let idx = (cur as u8) - b'A';
                        vars.sta_va[idx as usize] = arg;
                    } else {
                        return Err("stack is empty".to_string());
                    }
                } else if cur >= 'a' && cur <= 'z' {
                    if let Some(arg) = stack.pop() {
                        let idx = (cur as u8) - b'a';
                        vars.dyn_va[idx as usize] = arg;
                    } else {
                        return Err("stack is empty".to_string());
                    }
                } else {
                    return Err("bad variable name in %P".to_string());
                }
            }
            GetVar => {
                if cur >= 'A' && cur <= 'Z' {
                    let idx = (cur as u8) - b'A';
                    stack.push(vars.sta_va[idx as usize].clone());
                } else if cur >= 'a' && cur <= 'z' {
                    let idx = (cur as u8) - b'a';
                    stack.push(vars.dyn_va[idx as usize].clone());
                } else {
                    return Err("bad variable name in %g".to_string());
                }
            }
            CharConstant => {
                stack.push(Number(c as i32));
                state = CharClose;
            }
            CharClose => {
                if cur != '\'' {
                    return Err("malformed character constant".to_string());
                }
            }
            IntConstant(i) => {
                if cur == '}' {
                    stack.push(Number(i));
                    state = Nothing;
                } else if let Some(digit) = cur.to_digit(10) {
                    match i.checked_mul(10).and_then(|i_ten| i_ten.checked_add(digit as i32)) {
                        Some(i) => {
                            state = IntConstant(i);
                            old_state = Nothing;
                        }
                        None => return Err("int constant too large".to_string()),
                    }
                } else {
                    return Err("bad int constant".to_string());
                }
            }
            FormatPattern(ref mut flags, ref mut fstate) => {
                old_state = Nothing;
                match (*fstate, cur) {
                    (_, 'd') | (_, 'o') | (_, 'x') | (_, 'X') | (_, 's') => {
                        if let Some(arg) = stack.pop() {
                            let res = format(arg, FormatOp::from_char(cur), *flags)?;
                            output.extend(res.iter().cloned());
                            // will cause state to go to Nothing
                            old_state = FormatPattern(*flags, *fstate);
                        } else {
                            return Err("stack is empty".to_string());
                        }
                    }
                    (FormatState::Flags, '#') => {
                        flags.alternate = true;
                    }
                    (FormatState::Flags, '-') => {
                        flags.left = true;
                    }
                    (FormatState::Flags, '+') => {
                        flags.sign = true;
                    }
                    (FormatState::Flags, ' ') => {
                        flags.space = true;
                    }
                    (FormatState::Flags, '0'..='9') => {
                        flags.width = cur as usize - '0' as usize;
                        *fstate = FormatState::Width;
                    }
                    (FormatState::Flags, '.') => {
                        *fstate = FormatState::Precision;
                    }
                    (FormatState::Width, '0'..='9') => {
                        let old = flags.width;
                        flags.width = flags.width * 10 + (cur as usize - '0' as usize);
                        if flags.width < old {
                            return Err("format width overflow".to_string());
                        }
                    }
                    (FormatState::Width, '.') => {
                        *fstate = FormatState::Precision;
                    }
                    (FormatState::Precision, '0'..='9') => {
                        let old = flags.precision;
                        flags.precision = flags.precision * 10 + (cur as usize - '0' as usize);
                        if flags.precision < old {
                            return Err("format precision overflow".to_string());
                        }
                    }
                    _ => return Err("invalid format specifier".to_string()),
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
                        state = SeekIfElse(level - 1);
                    }
                } else if cur == 'e' && level == 0 {
                    state = Nothing;
                } else if cur == '?' {
                    state = SeekIfElse(level + 1);
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
                        state = SeekIfEnd(level - 1);
                    }
                } else if cur == '?' {
                    state = SeekIfEnd(level + 1);
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

#[derive(Copy, PartialEq, Clone)]
struct Flags {
    width: usize,
    precision: usize,
    alternate: bool,
    left: bool,
    sign: bool,
    space: bool,
}

impl Flags {
    fn new() -> Flags {
        Flags { width: 0, precision: 0, alternate: false, left: false, sign: false, space: false }
    }
}

#[derive(Copy, Clone)]
enum FormatOp {
    Digit,
    Octal,
    LowerHex,
    UpperHex,
    String,
}

impl FormatOp {
    fn from_char(c: char) -> FormatOp {
        match c {
            'd' => FormatOp::Digit,
            'o' => FormatOp::Octal,
            'x' => FormatOp::LowerHex,
            'X' => FormatOp::UpperHex,
            's' => FormatOp::String,
            _ => panic!("bad FormatOp char"),
        }
    }
}

fn format(val: Param, op: FormatOp, flags: Flags) -> Result<Vec<u8>, String> {
    let mut s = match val {
        Number(d) => {
            match op {
                FormatOp::Digit => {
                    if flags.sign {
                        format!("{:+01$}", d, flags.precision)
                    } else if d < 0 {
                        // C doesn't take sign into account in precision calculation.
                        format!("{:01$}", d, flags.precision + 1)
                    } else if flags.space {
                        format!(" {:01$}", d, flags.precision)
                    } else {
                        format!("{:01$}", d, flags.precision)
                    }
                }
                FormatOp::Octal => {
                    if flags.alternate {
                        // Leading octal zero counts against precision.
                        format!("0{:01$o}", d, flags.precision.saturating_sub(1))
                    } else {
                        format!("{:01$o}", d, flags.precision)
                    }
                }
                FormatOp::LowerHex => {
                    if flags.alternate && d != 0 {
                        format!("0x{:01$x}", d, flags.precision)
                    } else {
                        format!("{:01$x}", d, flags.precision)
                    }
                }
                FormatOp::UpperHex => {
                    if flags.alternate && d != 0 {
                        format!("0X{:01$X}", d, flags.precision)
                    } else {
                        format!("{:01$X}", d, flags.precision)
                    }
                }
                FormatOp::String => return Err("non-number on stack with %s".to_string()),
            }
            .into_bytes()
        }
    };
    if flags.width > s.len() {
        let n = flags.width - s.len();
        if flags.left {
            s.extend(repeat(b' ').take(n));
        } else {
            let mut s_ = Vec::with_capacity(flags.width);
            s_.extend(repeat(b' ').take(n));
            s_.extend(s.into_iter());
            s = s_;
        }
    }
    Ok(s)
}
