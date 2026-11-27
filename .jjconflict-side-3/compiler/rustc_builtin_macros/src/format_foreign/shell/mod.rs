use rustc_span::InnerSpan;

use super::StrCursor as Cur;

#[derive(Clone, PartialEq, Debug)]
pub(crate) enum Substitution<'a> {
    Ordinal(u8, (usize, usize)),
    Name(&'a str, (usize, usize)),
    Escape((usize, usize)),
}

impl ToString for Substitution<'_> {
    fn to_string(&self) -> String {
        match self {
            Substitution::Ordinal(n, _) => format!("${n}"),
            Substitution::Name(n, _) => format!("${n}"),
            Substitution::Escape(_) => "$$".into(),
        }
    }
}

impl Substitution<'_> {
    pub(crate) fn position(&self) -> InnerSpan {
        let (Self::Ordinal(_, pos) | Self::Name(_, pos) | Self::Escape(pos)) = self;
        InnerSpan::new(pos.0, pos.1)
    }

    fn set_position(&mut self, start: usize, end: usize) {
        let (Self::Ordinal(_, pos) | Self::Name(_, pos) | Self::Escape(pos)) = self;
        *pos = (start, end);
    }

    pub(crate) fn translate(&self) -> Result<String, Option<String>> {
        match self {
            Substitution::Ordinal(n, _) => Ok(format!("{{{}}}", n)),
            Substitution::Name(n, _) => Ok(format!("{{{}}}", n)),
            Substitution::Escape(_) => Err(None),
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
        (0, Some(self.s.len()))
    }
}

/// Parse the next substitution from the input string.
fn parse_next_substitution(s: &str) -> Option<(Substitution<'_>, &str)> {
    let at = {
        let start = s.find('$')?;
        match s[start + 1..].chars().next()? {
            '$' => return Some((Substitution::Escape((start, start + 2)), &s[start + 2..])),
            c @ '0'..='9' => {
                let n = (c as u8) - b'0';
                return Some((Substitution::Ordinal(n, (start, start + 2)), &s[start + 2..]));
            }
            _ => { /* fall-through */ }
        }

        Cur::new_at(s, start)
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
where
    F: FnMut(char) -> bool,
{
    loop {
        match cur.next_cp() {
            Some((c, next)) if pred(c) => {
                cur = next;
            }
            _ => return cur,
        }
    }
}

fn is_ident_head(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_tail(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

#[cfg(test)]
mod tests;
