// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! try_opt {
    ($e:expr) => {
        match $e {
            Some(v) => v,
            None => return None,
        }
    };
}

pub mod printf {
    use super::strcursor::StrCursor as Cur;

    /// Represents a single `printf`-style substitution.
    #[derive(Clone, Eq, PartialEq, Debug)]
    pub enum Substitution<'a> {
        /// A formatted output substitution.
        Format(Format<'a>),
        /// A literal `%%` escape.
        Escape,
    }

    impl<'a> Substitution<'a> {
        pub fn as_str(&self) -> &str {
            match *self {
                Substitution::Format(ref fmt) => fmt.span,
                Substitution::Escape => "%%",
            }
        }

        /// Translate this substitution into an equivalent Rust formatting directive.
        ///
        /// This ignores cases where the substitution does not have an exact equivalent, or where
        /// the substitution would be unnecessary.
        pub fn translate(&self) -> Option<String> {
            match *self {
                Substitution::Format(ref fmt) => fmt.translate(),
                Substitution::Escape => None,
            }
        }
    }

    #[derive(Clone, Eq, PartialEq, Debug)]
    /// A single `printf`-style formatting directive.
    pub struct Format<'a> {
        /// The entire original formatting directive.
        pub span: &'a str,
        /// The (1-based) parameter to be converted.
        pub parameter: Option<u16>,
        /// Formatting flags.
        pub flags: &'a str,
        /// Minimum width of the output.
        pub width: Option<Num>,
        /// Precision of the conversion.
        pub precision: Option<Num>,
        /// Length modifier for the conversion.
        pub length: Option<&'a str>,
        /// Type of parameter being converted.
        pub type_: &'a str,
    }

    impl<'a> Format<'a> {
        /// Translate this directive into an equivalent Rust formatting directive.
        ///
        /// Returns `None` in cases where the `printf` directive does not have an exact Rust
        /// equivalent, rather than guessing.
        pub fn translate(&self) -> Option<String> {
            use std::fmt::Write;

            let (c_alt, c_zero, c_left, c_plus) = {
                let mut c_alt = false;
                let mut c_zero = false;
                let mut c_left = false;
                let mut c_plus = false;
                for c in self.flags.chars() {
                    match c {
                        '#' => c_alt = true,
                        '0' => c_zero = true,
                        '-' => c_left = true,
                        '+' => c_plus = true,
                        _ => return None
                    }
                }
                (c_alt, c_zero, c_left, c_plus)
            };

            // Has a special form in Rust for numbers.
            let fill = if c_zero { Some("0") } else { None };

            let align = if c_left { Some("<") } else { None };

            // Rust doesn't have an equivalent to the `' '` flag.
            let sign = if c_plus { Some("+") } else { None };

            // Not *quite* the same, depending on the type...
            let alt = c_alt;

            let width = match self.width {
                Some(Num::Next) => {
                    // NOTE: Rust doesn't support this.
                    return None;
                }
                w @ Some(Num::Arg(_)) => w,
                w @ Some(Num::Num(_)) => w,
                None => None,
            };

            let precision = self.precision;

            // NOTE: although length *can* have an effect, we can't duplicate the effect in Rust, so
            // we just ignore it.

            let (type_, use_zero_fill, is_int) = match self.type_ {
                "d" | "i" | "u" => (None, true, true),
                "f" | "F" => (None, false, false),
                "s" | "c" => (None, false, false),
                "e" | "E" => (Some(self.type_), true, false),
                "x" | "X" | "o" => (Some(self.type_), true, true),
                "p" => (Some(self.type_), false, true),
                "g" => (Some("e"), true, false),
                "G" => (Some("E"), true, false),
                _ => return None,
            };

            let (fill, width, precision) = match (is_int, width, precision) {
                (true, Some(_), Some(_)) => {
                    // Rust can't duplicate this insanity.
                    return None;
                },
                (true, None, Some(p)) => (Some("0"), Some(p), None),
                (true, w, None) => (fill, w, None),
                (false, w, p) => (fill, w, p),
            };

            let align = match (self.type_, width.is_some(), align.is_some()) {
                ("s", true, false) => Some(">"),
                _ => align,
            };

            let (fill, zero_fill) = match (fill, use_zero_fill) {
                (Some("0"), true) => (None, true),
                (fill, _) => (fill, false),
            };

            let alt = match type_ {
                Some("x") | Some("X") => alt,
                _ => false,
            };

            let has_options = fill.is_some()
                || align.is_some()
                || sign.is_some()
                || alt
                || zero_fill
                || width.is_some()
                || precision.is_some()
                || type_.is_some()
                ;

            // Initialise with a rough guess.
            let cap = self.span.len() + if has_options { 2 } else { 0 };
            let mut s = String::with_capacity(cap);

            s.push_str("{");

            if let Some(arg) = self.parameter {
                try_opt!(write!(s, "{}", try_opt!(arg.checked_sub(1))).ok());
            }

            if has_options {
                s.push_str(":");

                let align = if let Some(fill) = fill {
                    s.push_str(fill);
                    align.or(Some(">"))
                } else {
                    align
                };

                if let Some(align) = align {
                    s.push_str(align);
                }

                if let Some(sign) = sign {
                    s.push_str(sign);
                }

                if alt {
                    s.push_str("#");
                }

                if zero_fill {
                    s.push_str("0");
                }

                if let Some(width) = width {
                    try_opt!(width.translate(&mut s).ok());
                }

                if let Some(precision) = precision {
                    s.push_str(".");
                    try_opt!(precision.translate(&mut s).ok());
                }

                if let Some(type_) = type_ {
                    s.push_str(type_);
                }
            }

            s.push_str("}");
            Some(s)
        }
    }

    /// A general number used in a `printf` formatting directive.
    #[derive(Copy, Clone, Eq, PartialEq, Debug)]
    pub enum Num {
        // The range of these values is technically bounded by `NL_ARGMAX`... but, at least for GNU
        // libc, it apparently has no real fixed limit.  A `u16` is used here on the basis that it
        // is *vanishingly* unlikely that *anyone* is going to try formatting something wider, or
        // with more precision, than 32 thousand positions which is so wide it couldn't possibly fit
        // on a screen.

        /// A specific, fixed value.
        Num(u16),
        /// The value is derived from a positional argument.
        Arg(u16),
        /// The value is derived from the "next" unconverted argument.
        Next,
    }

    impl Num {
        fn from_str(s: &str, arg: Option<&str>) -> Self {
            if let Some(arg) = arg {
                Num::Arg(arg.parse().expect(&format!("invalid format arg `{:?}`", arg)))
            } else if s == "*" {
                Num::Next
            } else {
                Num::Num(s.parse().expect(&format!("invalid format num `{:?}`", s)))
            }
        }

        fn translate(&self, s: &mut String) -> ::std::fmt::Result {
            use std::fmt::Write;
            match *self {
                Num::Num(n) => write!(s, "{}", n),
                Num::Arg(n) => {
                    let n = try!(n.checked_sub(1).ok_or(::std::fmt::Error));
                    write!(s, "{}$", n)
                },
                Num::Next => write!(s, "*"),
            }
        }
    }

    /// Returns an iterator over all substitutions in a given string.
    pub fn iter_subs(s: &str) -> Substitutions {
        Substitutions {
            s: s,
        }
    }

    /// Iterator over substitutions in a string.
    pub struct Substitutions<'a> {
        s: &'a str,
    }

    impl<'a> Iterator for Substitutions<'a> {
        type Item = Substitution<'a>;
        fn next(&mut self) -> Option<Self::Item> {
            match parse_next_substitution(self.s) {
                Some((sub, tail)) => {
                    self.s = tail;
                    Some(sub)
                },
                None => None,
            }
        }
    }

    enum State {
        Start,
        Flags,
        Width,
        WidthArg,
        Prec,
        PrecInner,
        Length,
        Type,
    }

    /// Parse the next substitution from the input string.
    pub fn parse_next_substitution(s: &str) -> Option<(Substitution, &str)> {
        use self::State::*;

        let at = {
            let start = try_opt!(s.find('%'));
            match s[start+1..].chars().next() {
                Some('%') => return Some((Substitution::Escape, &s[start+2..])),
                Some(_) => {/* fall-through */},
                None => return None,
            }

            Cur::new_at_start(&s[start..])
        };

        // This is meant to be a translation of the following regex:
        //
        // ```regex
        // (?x)
        // ^ %
        // (?: (?P<parameter> \d+) \$ )?
        // (?P<flags> [-+ 0\#']* )
        // (?P<width> \d+ | \* (?: (?P<widtha> \d+) \$ )? )?
        // (?: \. (?P<precision> \d+ | \* (?: (?P<precisiona> \d+) \$ )? ) )?
        // (?P<length>
        //     # Standard
        //     hh | h | ll | l | L | z | j | t
        //
        //     # Other
        //     | I32 | I64 | I | q
        // )?
        // (?P<type> . )
        // ```

        // Used to establish the full span at the end.
        let start = at;
        // The current position within the string.
        let mut at = try_opt!(at.at_next_cp());
        // `c` is the next codepoint, `next` is a cursor after it.
        let (mut c, mut next) = try_opt!(at.next_cp());

        // Update `at`, `c`, and `next`, exiting if we're out of input.
        macro_rules! move_to {
            ($cur:expr) => {
                {
                    at = $cur;
                    let (c_, next_) = try_opt!(at.next_cp());
                    c = c_;
                    next = next_;
                }
            };
        }

        // Constructs a result when parsing fails.
        //
        // Note: `move` used to capture copies of the cursors as they are *now*.
        let fallback = move || {
            return Some((
                Substitution::Format(Format {
                    span: start.slice_between(next).unwrap(),
                    parameter: None,
                    flags: "",
                    width: None,
                    precision: None,
                    length: None,
                    type_: at.slice_between(next).unwrap(),
                }),
                next.slice_after()
            ));
        };

        // Next parsing state.
        let mut state = Start;

        // Sadly, Rust isn't *quite* smart enough to know these *must* be initialised by the end.
        let mut parameter: Option<u16> = None;
        let mut flags: &str = "";
        let mut width: Option<Num> = None;
        let mut precision: Option<Num> = None;
        let mut length: Option<&str> = None;
        let mut type_: &str = "";
        let end: Cur;

        if let Start = state {
            match c {
                '1'...'9' => {
                    let end = at_next_cp_while(next, is_digit);
                    match end.next_cp() {
                        // Yes, this *is* the parameter.
                        Some(('$', end2)) => {
                            state = Flags;
                            parameter = Some(at.slice_between(end).unwrap().parse().unwrap());
                            move_to!(end2);
                        },
                        // Wait, no, actually, it's the width.
                        Some(_) => {
                            state = Prec;
                            parameter = None;
                            flags = "";
                            width = Some(Num::from_str(at.slice_between(end).unwrap(), None));
                            move_to!(end);
                        },
                        // It's invalid, is what it is.
                        None => return fallback(),
                    }
                },
                _ => {
                    state = Flags;
                    parameter = None;
                    move_to!(at);
                }
            }
        }

        if let Flags = state {
            let end = at_next_cp_while(at, is_flag);
            state = Width;
            flags = at.slice_between(end).unwrap();
            move_to!(end);
        }

        if let Width = state {
            match c {
                '*' => {
                    state = WidthArg;
                    move_to!(next);
                },
                '1' ... '9' => {
                    let end = at_next_cp_while(next, is_digit);
                    state = Prec;
                    width = Some(Num::from_str(at.slice_between(end).unwrap(), None));
                    move_to!(end);
                },
                _ => {
                    state = Prec;
                    width = None;
                    move_to!(at);
                }
            }
        }

        if let WidthArg = state {
            let end = at_next_cp_while(at, is_digit);
            match end.next_cp() {
                Some(('$', end2)) => {
                    state = Prec;
                    width = Some(Num::from_str("", Some(at.slice_between(end).unwrap())));
                    move_to!(end2);
                },
                _ => {
                    state = Prec;
                    width = Some(Num::Next);
                    move_to!(end);
                }
            }
        }

        if let Prec = state {
            match c {
                '.' => {
                    state = PrecInner;
                    move_to!(next);
                },
                _ => {
                    state = Length;
                    precision = None;
                    move_to!(at);
                }
            }
        }

        if let PrecInner = state {
            match c {
                '*' => {
                    let end = at_next_cp_while(next, is_digit);
                    match end.next_cp() {
                        Some(('$', end2)) => {
                            state = Length;
                            precision = Some(Num::from_str("*", next.slice_between(end)));
                            move_to!(end2);
                        },
                        _ => {
                            state = Length;
                            precision = Some(Num::Next);
                            move_to!(end);
                        }
                    }
                },
                '0' ... '9' => {
                    let end = at_next_cp_while(next, is_digit);
                    state = Length;
                    precision = Some(Num::from_str(at.slice_between(end).unwrap(), None));
                    move_to!(end);
                },
                _ => return fallback(),
            }
        }

        if let Length = state {
            let c1_next1 = next.next_cp();
            match (c, c1_next1) {
                ('h', Some(('h', next1)))
                | ('l', Some(('l', next1)))
                => {
                    state = Type;
                    length = Some(at.slice_between(next1).unwrap());
                    move_to!(next1);
                },

                ('h', _) | ('l', _) | ('L', _)
                | ('z', _) | ('j', _) | ('t', _)
                | ('q', _)
                => {
                    state = Type;
                    length = Some(at.slice_between(next).unwrap());
                    move_to!(next);
                },

                ('I', _) => {
                    let end = next.at_next_cp()
                        .and_then(|end| end.at_next_cp())
                        .map(|end| (next.slice_between(end).unwrap(), end));
                    let end = match end {
                        Some(("32", end)) => end,
                        Some(("64", end)) => end,
                        _ => next
                    };
                    state = Type;
                    length = Some(at.slice_between(end).unwrap());
                    move_to!(end);
                },

                _ => {
                    state = Type;
                    length = None;
                    move_to!(at);
                }
            }
        }

        if let Type = state {
            drop(c);
            type_ = at.slice_between(next).unwrap();

            // Don't use `move_to!` here, as we *can* be at the end of the input.
            at = next;
        }

        drop(c);
        drop(next);

        end = at;

        let f = Format {
            span: start.slice_between(end).unwrap(),
            parameter: parameter,
            flags: flags,
            width: width,
            precision: precision,
            length: length,
            type_: type_,
        };
        Some((Substitution::Format(f), end.slice_after()))
    }

    fn at_next_cp_while<F>(mut cur: Cur, mut pred: F) -> Cur
    where F: FnMut(char) -> bool {
        loop {
            match cur.next_cp() {
                Some((c, next)) => if pred(c) {
                    cur = next;
                } else {
                    return cur;
                },
                None => return cur,
            }
        }
    }

    fn is_digit(c: char) -> bool {
        match c {
            '0' ... '9' => true,
            _ => false
        }
    }

    fn is_flag(c: char) -> bool {
        match c {
            '0' | '-' | '+' | ' ' | '#' | '\'' => true,
            _ => false
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{
            Format as F,
            Num as N,
            Substitution as S,
            iter_subs,
            parse_next_substitution as pns,
        };

        macro_rules! assert_eq_pnsat {
            ($lhs:expr, $rhs:expr) => {
                assert_eq!(
                    pns($lhs).and_then(|(s, _)| s.translate()),
                    $rhs.map(<String as From<&str>>::from)
                )
            };
        }

        #[test]
        fn test_escape() {
            assert_eq!(pns("has no escapes"), None);
            assert_eq!(pns("has no escapes, either %"), None);
            assert_eq!(pns("*so* has a %% escape"), Some((S::Escape," escape")));
            assert_eq!(pns("%% leading escape"), Some((S::Escape, " leading escape")));
            assert_eq!(pns("trailing escape %%"), Some((S::Escape, "")));
        }

        #[test]
        fn test_parse() {
            macro_rules! assert_pns_eq_sub {
                ($in_:expr, {
                    $param:expr, $flags:expr,
                    $width:expr, $prec:expr, $len:expr, $type_:expr,
                }) => {
                    assert_eq!(
                        pns(concat!($in_, "!")),
                        Some((
                            S::Format(F {
                                span: $in_,
                                parameter: $param,
                                flags: $flags,
                                width: $width,
                                precision: $prec,
                                length: $len,
                                type_: $type_,
                            }),
                            "!"
                        ))
                    )
                };
            }

            assert_pns_eq_sub!("%!",
                { None, "", None, None, None, "!", });
            assert_pns_eq_sub!("%c",
                { None, "", None, None, None, "c", });
            assert_pns_eq_sub!("%s",
                { None, "", None, None, None, "s", });
            assert_pns_eq_sub!("%06d",
                { None, "0", Some(N::Num(6)), None, None, "d", });
            assert_pns_eq_sub!("%4.2f",
                { None, "", Some(N::Num(4)), Some(N::Num(2)), None, "f", });
            assert_pns_eq_sub!("%#x",
                { None, "#", None, None, None, "x", });
            assert_pns_eq_sub!("%-10s",
                { None, "-", Some(N::Num(10)), None, None, "s", });
            assert_pns_eq_sub!("%*s",
                { None, "", Some(N::Next), None, None, "s", });
            assert_pns_eq_sub!("%-10.*s",
                { None, "-", Some(N::Num(10)), Some(N::Next), None, "s", });
            assert_pns_eq_sub!("%-*.*s",
                { None, "-", Some(N::Next), Some(N::Next), None, "s", });
            assert_pns_eq_sub!("%.6i",
                { None, "", None, Some(N::Num(6)), None, "i", });
            assert_pns_eq_sub!("%+i",
                { None, "+", None, None, None, "i", });
            assert_pns_eq_sub!("%08X",
                { None, "0", Some(N::Num(8)), None, None, "X", });
            assert_pns_eq_sub!("%lu",
                { None, "", None, None, Some("l"), "u", });
            assert_pns_eq_sub!("%Iu",
                { None, "", None, None, Some("I"), "u", });
            assert_pns_eq_sub!("%I32u",
                { None, "", None, None, Some("I32"), "u", });
            assert_pns_eq_sub!("%I64u",
                { None, "", None, None, Some("I64"), "u", });
            assert_pns_eq_sub!("%'d",
                { None, "'", None, None, None, "d", });
            assert_pns_eq_sub!("%10s",
                { None, "", Some(N::Num(10)), None, None, "s", });
            assert_pns_eq_sub!("%-10.10s",
                { None, "-", Some(N::Num(10)), Some(N::Num(10)), None, "s", });
            assert_pns_eq_sub!("%1$d",
                { Some(1), "", None, None, None, "d", });
            assert_pns_eq_sub!("%2$.*3$d",
                { Some(2), "", None, Some(N::Arg(3)), None, "d", });
            assert_pns_eq_sub!("%1$*2$.*3$d",
                { Some(1), "", Some(N::Arg(2)), Some(N::Arg(3)), None, "d", });
            assert_pns_eq_sub!("%-8ld",
                { None, "-", Some(N::Num(8)), None, Some("l"), "d", });
        }

        #[test]
        fn test_iter() {
            let s = "The %d'th word %% is: `%.*s` %!\n";
            let subs: Vec<_> = iter_subs(s).map(|sub| sub.translate()).collect();
            assert_eq!(
                subs.iter().map(|ms| ms.as_ref().map(|s| &s[..])).collect::<Vec<_>>(),
                vec![Some("{}"), None, Some("{:.*}"), None]
            );
        }

        /// Check that the translations are what we expect.
        #[test]
        fn test_trans() {
            assert_eq_pnsat!("%c", Some("{}"));
            assert_eq_pnsat!("%d", Some("{}"));
            assert_eq_pnsat!("%u", Some("{}"));
            assert_eq_pnsat!("%x", Some("{:x}"));
            assert_eq_pnsat!("%X", Some("{:X}"));
            assert_eq_pnsat!("%e", Some("{:e}"));
            assert_eq_pnsat!("%E", Some("{:E}"));
            assert_eq_pnsat!("%f", Some("{}"));
            assert_eq_pnsat!("%g", Some("{:e}"));
            assert_eq_pnsat!("%G", Some("{:E}"));
            assert_eq_pnsat!("%s", Some("{}"));
            assert_eq_pnsat!("%p", Some("{:p}"));

            assert_eq_pnsat!("%06d",        Some("{:06}"));
            assert_eq_pnsat!("%4.2f",       Some("{:4.2}"));
            assert_eq_pnsat!("%#x",         Some("{:#x}"));
            assert_eq_pnsat!("%-10s",       Some("{:<10}"));
            assert_eq_pnsat!("%*s",         None);
            assert_eq_pnsat!("%-10.*s",     Some("{:<10.*}"));
            assert_eq_pnsat!("%-*.*s",      None);
            assert_eq_pnsat!("%.6i",        Some("{:06}"));
            assert_eq_pnsat!("%+i",         Some("{:+}"));
            assert_eq_pnsat!("%08X",        Some("{:08X}"));
            assert_eq_pnsat!("%lu",         Some("{}"));
            assert_eq_pnsat!("%Iu",         Some("{}"));
            assert_eq_pnsat!("%I32u",       Some("{}"));
            assert_eq_pnsat!("%I64u",       Some("{}"));
            assert_eq_pnsat!("%'d",         None);
            assert_eq_pnsat!("%10s",        Some("{:>10}"));
            assert_eq_pnsat!("%-10.10s",    Some("{:<10.10}"));
            assert_eq_pnsat!("%1$d",        Some("{0}"));
            assert_eq_pnsat!("%2$.*3$d",    Some("{1:02$}"));
            assert_eq_pnsat!("%1$*2$.*3$s", Some("{0:>1$.2$}"));
            assert_eq_pnsat!("%-8ld",       Some("{:<8}"));
        }
    }
}

pub mod shell {
    use super::strcursor::StrCursor as Cur;

    #[derive(Clone, Eq, PartialEq, Debug)]
    pub enum Substitution<'a> {
        Ordinal(u8),
        Name(&'a str),
        Escape,
    }

    impl<'a> Substitution<'a> {
        pub fn as_str(&self) -> String {
            match *self {
                Substitution::Ordinal(n) => format!("${}", n),
                Substitution::Name(n) => format!("${}", n),
                Substitution::Escape => "$$".into(),
            }
        }

        pub fn translate(&self) -> Option<String> {
            match *self {
                Substitution::Ordinal(n) => Some(format!("{{{}}}", n)),
                Substitution::Name(n) => Some(format!("{{{}}}", n)),
                Substitution::Escape => None,
            }
        }
    }

    /// Returns an iterator over all substitutions in a given string.
    pub fn iter_subs(s: &str) -> Substitutions {
        Substitutions {
            s: s,
        }
    }

    /// Iterator over substitutions in a string.
    pub struct Substitutions<'a> {
        s: &'a str,
    }

    impl<'a> Iterator for Substitutions<'a> {
        type Item = Substitution<'a>;
        fn next(&mut self) -> Option<Self::Item> {
            match parse_next_substitution(self.s) {
                Some((sub, tail)) => {
                    self.s = tail;
                    Some(sub)
                },
                None => None,
            }
        }
    }

    /// Parse the next substitution from the input string.
    pub fn parse_next_substitution(s: &str) -> Option<(Substitution, &str)> {
        let at = {
            let start = try_opt!(s.find('$'));
            match s[start+1..].chars().next() {
                Some('$') => return Some((Substitution::Escape, &s[start+2..])),
                Some(c @ '0' ... '9') => {
                    let n = (c as u8) - b'0';
                    return Some((Substitution::Ordinal(n), &s[start+2..]));
                },
                Some(_) => {/* fall-through */},
                None => return None,
            }

            Cur::new_at_start(&s[start..])
        };

        let at = try_opt!(at.at_next_cp());
        match at.next_cp() {
            Some((c, inner)) => {
                if !is_ident_head(c) {
                    None
                } else {
                    let end = at_next_cp_while(inner, is_ident_tail);
                    Some((Substitution::Name(at.slice_between(end).unwrap()), end.slice_after()))
                }
            },
            _ => None
        }
    }

    fn at_next_cp_while<F>(mut cur: Cur, mut pred: F) -> Cur
    where F: FnMut(char) -> bool {
        loop {
            match cur.next_cp() {
                Some((c, next)) => if pred(c) {
                    cur = next;
                } else {
                    return cur;
                },
                None => return cur,
            }
        }
    }

    fn is_ident_head(c: char) -> bool {
        match c {
            'a' ... 'z' | 'A' ... 'Z' | '_' => true,
            _ => false
        }
    }

    fn is_ident_tail(c: char) -> bool {
        match c {
            '0' ... '9' => true,
            c => is_ident_head(c)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{
            Substitution as S,
            parse_next_substitution as pns,
        };

        macro_rules! assert_eq_pnsat {
            ($lhs:expr, $rhs:expr) => {
                assert_eq!(
                    pns($lhs).and_then(|(f, _)| f.translate()),
                    $rhs.map(<String as From<&str>>::from)
                )
            };
        }

        #[test]
        fn test_escape() {
            assert_eq!(pns("has no escapes"), None);
            assert_eq!(pns("has no escapes, either $"), None);
            assert_eq!(pns("*so* has a $$ escape"), Some((S::Escape, " escape")));
            assert_eq!(pns("$$ leading escape"), Some((S::Escape, " leading escape")));
            assert_eq!(pns("trailing escape $$"), Some((S::Escape, "")));
        }

        #[test]
        fn test_parse() {
            macro_rules! assert_pns_eq_sub {
                ($in_:expr, $kind:ident($arg:expr)) => {
                    assert_eq!(pns(concat!($in_, "!")), Some((S::$kind($arg.into()), "!")))
                };
            }

            assert_pns_eq_sub!("$0", Ordinal(0));
            assert_pns_eq_sub!("$1", Ordinal(1));
            assert_pns_eq_sub!("$9", Ordinal(9));
            assert_pns_eq_sub!("$N", Name("N"));
            assert_pns_eq_sub!("$NAME", Name("NAME"));
        }

        #[test]
        fn test_iter() {
            use super::iter_subs;
            let s = "The $0'th word $$ is: `$WORD` $!\n";
            let subs: Vec<_> = iter_subs(s).map(|sub| sub.translate()).collect();
            assert_eq!(
                subs.iter().map(|ms| ms.as_ref().map(|s| &s[..])).collect::<Vec<_>>(),
                vec![Some("{0}"), None, Some("{WORD}")]
            );
        }

        #[test]
        fn test_trans() {
            assert_eq_pnsat!("$0", Some("{0}"));
            assert_eq_pnsat!("$9", Some("{9}"));
            assert_eq_pnsat!("$1", Some("{1}"));
            assert_eq_pnsat!("$10", Some("{1}"));
            assert_eq_pnsat!("$stuff", Some("{stuff}"));
            assert_eq_pnsat!("$NAME", Some("{NAME}"));
            assert_eq_pnsat!("$PREFIX/bin", Some("{PREFIX}"));
        }

    }
}

mod strcursor {
    use std;

    pub struct StrCursor<'a> {
        s: &'a str,
        at: usize,
    }

    impl<'a> StrCursor<'a> {
        pub fn new_at_start(s: &'a str) -> StrCursor<'a> {
            StrCursor {
                s: s,
                at: 0,
            }
        }

        pub fn at_next_cp(mut self) -> Option<StrCursor<'a>> {
            match self.try_seek_right_cp() {
                true => Some(self),
                false => None
            }
        }

        pub fn next_cp(mut self) -> Option<(char, StrCursor<'a>)> {
            let cp = match self.cp_after() {
                Some(cp) => cp,
                None => return None,
            };
            self.seek_right(cp.len_utf8());
            Some((cp, self))
        }

        fn slice_before(&self) -> &'a str {
            &self.s[0..self.at]
        }

        pub fn slice_after(&self) -> &'a str {
            &self.s[self.at..]
        }

        pub fn slice_between(&self, until: StrCursor<'a>) -> Option<&'a str> {
            if !str_eq_literal(self.s, until.s) {
                None
            } else {
                use std::cmp::{max, min};
                let beg = min(self.at, until.at);
                let end = max(self.at, until.at);
                Some(&self.s[beg..end])
            }
        }

        fn cp_after(&self) -> Option<char> {
            self.slice_after().chars().next()
        }

        fn try_seek_right_cp(&mut self) -> bool {
            match self.slice_after().chars().next() {
                Some(c) => {
                    self.at += c.len_utf8();
                    true
                },
                None => false,
            }
        }

        fn seek_right(&mut self, bytes: usize) {
            self.at += bytes;
        }
    }

    impl<'a> Copy for StrCursor<'a> {}

    impl<'a> Clone for StrCursor<'a> {
        fn clone(&self) -> StrCursor<'a> {
            *self
        }
    }

    impl<'a> std::fmt::Debug for StrCursor<'a> {
        fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
            write!(fmt, "StrCursor({:?} | {:?})", self.slice_before(), self.slice_after())
        }
    }

    fn str_eq_literal(a: &str, b: &str) -> bool {
        a.as_bytes().as_ptr() == b.as_bytes().as_ptr()
            && a.len() == b.len()
    }
}
