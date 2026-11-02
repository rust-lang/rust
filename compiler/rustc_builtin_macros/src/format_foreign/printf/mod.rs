use rustc_span::InnerSpan;

use super::StrCursor as Cur;

/// Represents a single `printf`-style substitution.
#[derive(Clone, PartialEq, Debug)]
pub(crate) enum Substitution<'a> {
    /// A formatted output substitution with its internal byte offset.
    Format(Format<'a>),
    /// A literal `%%` escape, with its start and end indices.
    Escape((usize, usize)),
}

impl ToString for Substitution<'_> {
    fn to_string(&self) -> String {
        match self {
            Substitution::Format(fmt) => fmt.span.into(),
            Substitution::Escape(_) => "%%".into(),
        }
    }
}

impl Substitution<'_> {
    pub(crate) fn position(&self) -> InnerSpan {
        match self {
            Substitution::Format(fmt) => fmt.position,
            &Substitution::Escape((start, end)) => InnerSpan::new(start, end),
        }
    }

    pub(crate) fn set_position(&mut self, start: usize, end: usize) {
        match self {
            Substitution::Format(fmt) => fmt.position = InnerSpan::new(start, end),
            Substitution::Escape(pos) => *pos = (start, end),
        }
    }

    /// Translate this substitution into an equivalent Rust formatting directive.
    ///
    /// This ignores cases where the substitution does not have an exact equivalent, or where
    /// the substitution would be unnecessary.
    pub(crate) fn translate(&self) -> Result<String, Option<String>> {
        match self {
            Substitution::Format(fmt) => fmt.translate(),
            Substitution::Escape(_) => Err(None),
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
/// A single `printf`-style formatting directive.
pub(crate) struct Format<'a> {
    /// The entire original formatting directive.
    span: &'a str,
    /// The (1-based) parameter to be converted.
    parameter: Option<u16>,
    /// Formatting flags.
    flags: &'a str,
    /// Minimum width of the output.
    width: Option<Num>,
    /// Precision of the conversion.
    precision: Option<Num>,
    /// Length modifier for the conversion.
    length: Option<&'a str>,
    /// Type of parameter being converted.
    type_: &'a str,
    /// Byte offset for the start and end of this formatting directive.
    position: InnerSpan,
}

impl Format<'_> {
    /// Translate this directive into an equivalent Rust formatting directive.
    ///
    /// Returns `Err` in cases where the `printf` directive does not have an exact Rust
    /// equivalent, rather than guessing.
    pub(crate) fn translate(&self) -> Result<String, Option<String>> {
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
                    _ => {
                        return Err(Some(format!("the flag `{c}` is unknown or unsupported")));
                    }
                }
            }
            (c_alt, c_zero, c_left, c_plus)
        };

        // Has a special form in Rust for numbers.
        let fill = c_zero.then_some("0");

        let align = c_left.then_some("<");

        // Rust doesn't have an equivalent to the `' '` flag.
        let sign = c_plus.then_some("+");

        // Not *quite* the same, depending on the type...
        let alt = c_alt;

        let width = match self.width {
            Some(Num::Next) => {
                // NOTE: Rust doesn't support this.
                return Err(Some(
                    "you have to use a positional or named parameter for the width".to_string(),
                ));
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
            _ => {
                return Err(Some(format!(
                    "the conversion specifier `{}` is unknown or unsupported",
                    self.type_
                )));
            }
        };

        let (fill, width, precision) = match (is_int, width, precision) {
            (true, Some(_), Some(_)) => {
                // Rust can't duplicate this insanity.
                return Err(Some(
                    "width and precision cannot both be specified for integer conversions"
                        .to_string(),
                ));
            }
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
            Some("x" | "X") => alt,
            _ => false,
        };

        let has_options = fill.is_some()
            || align.is_some()
            || sign.is_some()
            || alt
            || zero_fill
            || width.is_some()
            || precision.is_some()
            || type_.is_some();

        // Initialise with a rough guess.
        let cap = self.span.len() + if has_options { 2 } else { 0 };
        let mut s = String::with_capacity(cap);

        s.push('{');

        if let Some(arg) = self.parameter {
            write!(s, "{}", arg.checked_sub(1).ok_or(None)?).map_err(|_| None)?;
        }

        if has_options {
            s.push(':');

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
                s.push('#');
            }

            if zero_fill {
                s.push('0');
            }

            if let Some(width) = width {
                width.translate(&mut s).map_err(|_| None)?;
            }

            if let Some(precision) = precision {
                s.push('.');
                precision.translate(&mut s).map_err(|_| None)?;
            }

            if let Some(type_) = type_ {
                s.push_str(type_);
            }
        }

        s.push('}');
        Ok(s)
    }
}

/// A general number used in a `printf` formatting directive.
#[derive(Copy, Clone, PartialEq, Debug)]
enum Num {
    // The range of these values is technically bounded by `NL_ARGMAX`... but, at least for GNU
    // libc, it apparently has no real fixed limit. A `u16` is used here on the basis that it
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
    fn from_str(s: &str, arg: Option<&str>) -> Option<Self> {
        if let Some(arg) = arg {
            arg.parse().ok().map(Num::Arg)
        } else if s == "*" {
            Some(Num::Next)
        } else {
            s.parse().ok().map(Num::Num)
        }
    }

    fn translate(&self, s: &mut String) -> std::fmt::Result {
        use std::fmt::Write;
        match *self {
            Num::Num(n) => write!(s, "{n}"),
            Num::Arg(n) => {
                let n = n.checked_sub(1).ok_or(std::fmt::Error)?;
                write!(s, "{n}$")
            }
            Num::Next => write!(s, "*"),
        }
    }
}

/// Returns an iterator over all substitutions in a given string.
pub(crate) fn iter_subs(s: &str, start_pos: usize) -> Substitutions<'_> {
    Substitutions { s, pos: start_pos }
}

/// Iterator over substitutions in a string.
pub(crate) struct Substitutions<'a> {
    s: &'a str,
    pos: usize,
}

impl<'a> Iterator for Substitutions<'a> {
    type Item = Substitution<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        let (mut sub, tail) = parse_next_substitution(self.s)?;
        self.s = tail;
        let InnerSpan { start, end } = sub.position();
        sub.set_position(start + self.pos, end + self.pos);
        self.pos += end;
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
fn parse_next_substitution(s: &str) -> Option<(Substitution<'_>, &str)> {
    use self::State::*;

    let at = {
        let start = s.find('%')?;
        if let '%' = s[start + 1..].chars().next()? {
            return Some((Substitution::Escape((start, start + 2)), &s[start + 2..]));
        }

        Cur::new_at(s, start)
    };

    // This is meant to be a translation of the following regex:
    //
    // ```regex
    // (?x)
    // ^ %
    // (?: (?Box<parameter> \d+) \$ )?
    // (?Box<flags> [-+ 0\#']* )
    // (?Box<width> \d+ | \* (?: (?Box<widtha> \d+) \$ )? )?
    // (?: \. (?Box<precision> \d+ | \* (?: (?Box<precisiona> \d+) \$ )? ) )?
    // (?Box<length>
    //     # Standard
    //     hh | h | ll | l | L | z | j | t
    //
    //     # Other
    //     | I32 | I64 | I | q
    // )?
    // (?Box<type> . )
    // ```

    // Used to establish the full span at the end.
    let start = at;
    // The current position within the string.
    let mut at = at.at_next_cp()?;
    // `c` is the next codepoint, `next` is a cursor after it.
    let (mut c, mut next) = at.next_cp()?;

    // Update `at`, `c`, and `next`, exiting if we're out of input.
    macro_rules! move_to {
        ($cur:expr) => {{
            at = $cur;
            let (c_, next_) = at.next_cp()?;
            c = c_;
            next = next_;
        }};
    }

    // Constructs a result when parsing fails.
    //
    // Note: `move` used to capture copies of the cursors as they are *now*.
    let fallback = move || {
        Some((
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
            next.slice_after(),
        ))
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
                let end = at_next_cp_while(next, char::is_ascii_digit);
                match end.next_cp() {
                    // Yes, this *is* the parameter.
                    Some(('$', end2)) => {
                        state = Flags;
                        parameter = at.slice_between(end).unwrap().parse().ok();
                        move_to!(end2);
                    }
                    // Wait, no, actually, it's the width.
                    Some(_) => {
                        state = Prec;
                        parameter = None;
                        flags = "";
                        width = at.slice_between(end).and_then(|num| Num::from_str(num, None));
                        if width.is_none() {
                            return fallback();
                        }
                        move_to!(end);
                    }
                    // It's invalid, is what it is.
                    None => return fallback(),
                }
            }
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
            }
            '1'..='9' => {
                let end = at_next_cp_while(next, char::is_ascii_digit);
                state = Prec;
                width = at.slice_between(end).and_then(|num| Num::from_str(num, None));
                if width.is_none() {
                    return fallback();
                }
                move_to!(end);
            }
            _ => {
                state = Prec;
                width = None;
                move_to!(at);
            }
        }
    }

    if let WidthArg = state {
        let end = at_next_cp_while(at, char::is_ascii_digit);
        match end.next_cp() {
            Some(('$', end2)) => {
                state = Prec;
                width = Num::from_str("", at.slice_between(end));
                move_to!(end2);
            }
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
            }
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
                let end = at_next_cp_while(next, char::is_ascii_digit);
                match end.next_cp() {
                    Some(('$', end2)) => {
                        state = Length;
                        precision = Num::from_str("*", next.slice_between(end));
                        move_to!(end2);
                    }
                    _ => {
                        state = Length;
                        precision = Some(Num::Next);
                        move_to!(end);
                    }
                }
            }
            '0'..='9' => {
                let end = at_next_cp_while(next, char::is_ascii_digit);
                state = Length;
                precision = at.slice_between(end).and_then(|num| Num::from_str(num, None));
                move_to!(end);
            }
            _ => return fallback(),
        }
    }

    if let Length = state {
        let c1_next1 = next.next_cp();
        match (c, c1_next1) {
            ('h', Some(('h', next1))) | ('l', Some(('l', next1))) => {
                state = Type;
                length = Some(at.slice_between(next1).unwrap());
                move_to!(next1);
            }

            ('h' | 'l' | 'L' | 'z' | 'j' | 't' | 'q', _) => {
                state = Type;
                length = Some(at.slice_between(next).unwrap());
                move_to!(next);
            }

            ('I', _) => {
                let end = next
                    .at_next_cp()
                    .and_then(|end| end.at_next_cp())
                    .map(|end| (next.slice_between(end).unwrap(), end));
                let end = match end {
                    Some(("32" | "64", end)) => end,
                    _ => next,
                };
                state = Type;
                length = Some(at.slice_between(end).unwrap());
                move_to!(end);
            }

            _ => {
                state = Type;
                length = None;
                move_to!(at);
            }
        }
    }

    if let Type = state {
        type_ = at.slice_between(next).unwrap();

        // Don't use `move_to!` here, as we *can* be at the end of the input.
        at = next;
    }

    let _ = c; // to avoid never used value

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
where
    F: FnMut(&char) -> bool,
{
    loop {
        match cur.next_cp() {
            Some((c, next)) if pred(&c) => {
                cur = next;
            }
            _ => return cur,
        }
    }
}

fn is_flag(c: &char) -> bool {
    matches!(c, '0' | '-' | '+' | ' ' | '#' | '\'')
}

#[cfg(test)]
mod tests;
