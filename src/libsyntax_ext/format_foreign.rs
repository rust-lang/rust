pub mod printf {
    use super::strcursor::StrCursor as Cur;
    use syntax_pos::InnerSpan;

    /// Represents a single `printf`-style substitution.
    #[derive(Clone, PartialEq, Debug)]
    pub enum Substitution<'a> {
        /// A formatted output substitution with its internal byte offset.
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

        pub fn position(&self) -> Option<InnerSpan> {
            match *self {
                Substitution::Format(ref fmt) => Some(fmt.position),
                _ => None,
            }
        }

        pub fn set_position(&mut self, start: usize, end: usize) {
            match self {
                Substitution::Format(ref mut fmt) => {
                    fmt.position = InnerSpan::new(start, end);
                }
                _ => {}
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

    #[derive(Clone, PartialEq, Debug)]
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
        /// Byte offset for the start and end of this formatting directive.
        pub position: InnerSpan,
    }

    impl Format<'_> {
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
                write!(s, "{}", arg.checked_sub(1)?).ok()?;
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
                    width.translate(&mut s).ok()?;
                }

                if let Some(precision) = precision {
                    s.push_str(".");
                    precision.translate(&mut s).ok()?;
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
    #[derive(Copy, Clone, PartialEq, Debug)]
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
                Num::Arg(arg.parse().unwrap_or_else(|_| panic!("invalid format arg `{:?}`", arg)))
            } else if s == "*" {
                Num::Next
            } else {
                Num::Num(s.parse().unwrap_or_else(|_| panic!("invalid format num `{:?}`", s)))
            }
        }

        fn translate(&self, s: &mut String) -> std::fmt::Result {
            use std::fmt::Write;
            match *self {
                Num::Num(n) => write!(s, "{}", n),
                Num::Arg(n) => {
                    let n = n.checked_sub(1).ok_or(std::fmt::Error)?;
                    write!(s, "{}$", n)
                },
                Num::Next => write!(s, "*"),
            }
        }
    }

    /// Returns an iterator over all substitutions in a given string.
    pub fn iter_subs(s: &str, start_pos: usize) -> Substitutions<'_> {
        Substitutions {
            s,
            pos: start_pos,
        }
    }

    /// Iterator over substitutions in a string.
    pub struct Substitutions<'a> {
        s: &'a str,
        pos: usize,
    }

    impl<'a> Iterator for Substitutions<'a> {
        type Item = Substitution<'a>;
        fn next(&mut self) -> Option<Self::Item> {
            let (mut sub, tail) = parse_next_substitution(self.s)?;
            self.s = tail;
            match sub {
                Substitution::Format(_) => if let Some(inner_span) = sub.position() {
                    sub.set_position(inner_span.start + self.pos, inner_span.end + self.pos);
                    self.pos += inner_span.end;
                }
                Substitution::Escape => self.pos += 2,
            }
            Some(sub)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            // Substitutions are at least 2 characters long.
            (0, Some(self.s.len() / 2))
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
    pub fn parse_next_substitution(s: &str) -> Option<(Substitution<'_>, &str)> {
        use self::State::*;

        let at = {
            let start = s.find('%')?;
            match s[start+1..].chars().next()? {
                '%' => return Some((Substitution::Escape, &s[start+2..])),
                _ => {/* fall-through */},
            }

            Cur::new_at(&s[..], start)
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
        let mut at = at.at_next_cp()?;
        // `c` is the next codepoint, `next` is a cursor after it.
        let (mut c, mut next) = at.next_cp()?;

        // Update `at`, `c`, and `next`, exiting if we're out of input.
        macro_rules! move_to {
            ($cur:expr) => {
                {
                    at = $cur;
                    let (c_, next_) = at.next_cp()?;
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
                    position: InnerSpan::new(start.at, next.at),
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
        let end: Cur<'_>;

        if let Start = state {
            match c {
                '1'..='9' => {
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
                '1' ..= '9' => {
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
                '0' ..= '9' => {
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
        let position = InnerSpan::new(start.at, end.at);

        let f = Format {
            span: start.slice_between(end).unwrap(),
            parameter,
            flags,
            width,
            precision,
            length,
            type_,
            position,
        };
        Some((Substitution::Format(f), end.slice_after()))
    }

    fn at_next_cp_while<F>(mut cur: Cur<'_>, mut pred: F) -> Cur<'_>
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
            '0' ..= '9' => true,
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
    mod tests;
}

pub mod shell {
    use super::strcursor::StrCursor as Cur;
    use syntax_pos::InnerSpan;

    #[derive(Clone, PartialEq, Debug)]
    pub enum Substitution<'a> {
        Ordinal(u8, (usize, usize)),
        Name(&'a str, (usize, usize)),
        Escape((usize, usize)),
    }

    impl Substitution<'_> {
        pub fn as_str(&self) -> String {
            match self {
                Substitution::Ordinal(n, _) => format!("${}", n),
                Substitution::Name(n, _) => format!("${}", n),
                Substitution::Escape(_) => "$$".into(),
            }
        }

        pub fn position(&self) -> Option<InnerSpan> {
            match self {
                Substitution::Ordinal(_, pos) |
                Substitution::Name(_, pos) |
                Substitution::Escape(pos) => Some(InnerSpan::new(pos.0, pos.1)),
            }
        }

        pub fn set_position(&mut self, start: usize, end: usize) {
            match self {
                Substitution::Ordinal(_, ref mut pos) |
                Substitution::Name(_, ref mut pos) |
                Substitution::Escape(ref mut pos) => *pos = (start, end),
            }
        }

        pub fn translate(&self) -> Option<String> {
            match *self {
                Substitution::Ordinal(n, _) => Some(format!("{{{}}}", n)),
                Substitution::Name(n, _) => Some(format!("{{{}}}", n)),
                Substitution::Escape(_) => None,
            }
        }
    }

    /// Returns an iterator over all substitutions in a given string.
    pub fn iter_subs(s: &str, start_pos: usize) -> Substitutions<'_> {
        Substitutions {
            s,
            pos: start_pos,
        }
    }

    /// Iterator over substitutions in a string.
    pub struct Substitutions<'a> {
        s: &'a str,
        pos: usize,
    }

    impl<'a> Iterator for Substitutions<'a> {
        type Item = Substitution<'a>;
        fn next(&mut self) -> Option<Self::Item> {
            match parse_next_substitution(self.s) {
                Some((mut sub, tail)) => {
                    self.s = tail;
                    if let Some(InnerSpan { start, end }) = sub.position() {
                        sub.set_position(start + self.pos, end + self.pos);
                        self.pos += end;
                    }
                    Some(sub)
                },
                None => None,
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (0, Some(self.s.len()))
        }
    }

    /// Parse the next substitution from the input string.
    pub fn parse_next_substitution(s: &str) -> Option<(Substitution<'_>, &str)> {
        let at = {
            let start = s.find('$')?;
            match s[start+1..].chars().next()? {
                '$' => return Some((Substitution::Escape((start, start+2)), &s[start+2..])),
                c @ '0' ..= '9' => {
                    let n = (c as u8) - b'0';
                    return Some((Substitution::Ordinal(n, (start, start+2)), &s[start+2..]));
                },
                _ => {/* fall-through */},
            }

            Cur::new_at(&s[..], start)
        };

        let at = at.at_next_cp()?;
        let (c, inner) = at.next_cp()?;

        if !is_ident_head(c) {
            None
        } else {
            let end = at_next_cp_while(inner, is_ident_tail);
            let slice = at.slice_between(end).unwrap();
            let start = at.at - 1;
            let end_pos = at.at + slice.len();
            Some((Substitution::Name(slice, (start, end_pos)), end.slice_after()))
        }
    }

    fn at_next_cp_while<F>(mut cur: Cur<'_>, mut pred: F) -> Cur<'_>
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
            'a' ..= 'z' | 'A' ..= 'Z' | '_' => true,
            _ => false
        }
    }

    fn is_ident_tail(c: char) -> bool {
        match c {
            '0' ..= '9' => true,
            c => is_ident_head(c)
        }
    }

    #[cfg(test)]
    mod tests;
}

mod strcursor {
    pub struct StrCursor<'a> {
        s: &'a str,
        pub at: usize,
    }

    impl<'a> StrCursor<'a> {
        pub fn new_at(s: &'a str, at: usize) -> StrCursor<'a> {
            StrCursor {
                s,
                at,
            }
        }

        pub fn at_next_cp(mut self) -> Option<StrCursor<'a>> {
            match self.try_seek_right_cp() {
                true => Some(self),
                false => None
            }
        }

        pub fn next_cp(mut self) -> Option<(char, StrCursor<'a>)> {
            let cp = self.cp_after()?;
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

    impl Copy for StrCursor<'_> {}

    impl<'a> Clone for StrCursor<'a> {
        fn clone(&self) -> StrCursor<'a> {
            *self
        }
    }

    impl std::fmt::Debug for StrCursor<'_> {
        fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(fmt, "StrCursor({:?} | {:?})", self.slice_before(), self.slice_after())
        }
    }

    fn str_eq_literal(a: &str, b: &str) -> bool {
        a.as_bytes().as_ptr() == b.as_bytes().as_ptr()
            && a.len() == b.len()
    }
}
