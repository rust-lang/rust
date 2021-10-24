/// Dealing with sting indices can be hard, this struct ensures that both the
/// character and byte index are provided for correct indexing.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct StrIndex {
    pub char_index: usize,
    pub byte_index: usize,
}

impl StrIndex {
    pub fn new(char_index: usize, byte_index: usize) -> Self {
        Self { char_index, byte_index }
    }
}

/// Returns the index of the character after the first camel-case component of `s`.
///
/// ```
/// assert_eq!(camel_case_until("AbcDef"), StrIndex::new(6, 6));
/// assert_eq!(camel_case_until("ABCD"), StrIndex::new(0, 0));
/// assert_eq!(camel_case_until("AbcDD"), StrIndex::new(3, 3));
/// assert_eq!(camel_case_until("Abc\u{f6}\u{f6}DD"), StrIndex::new(5, 7));
/// ```
#[must_use]
pub fn camel_case_until(s: &str) -> StrIndex {
    let mut iter = s.char_indices().enumerate();
    if let Some((_char_index, (_, first))) = iter.next() {
        if !first.is_uppercase() {
            return StrIndex::new(0, 0);
        }
    } else {
        return StrIndex::new(0, 0);
    }
    let mut up = true;
    let mut last_index = StrIndex::new(0, 0);
    for (char_index, (byte_index, c)) in iter {
        if up {
            if c.is_lowercase() {
                up = false;
            } else {
                return last_index;
            }
        } else if c.is_uppercase() {
            up = true;
            last_index.byte_index = byte_index;
            last_index.char_index = char_index;
        } else if !c.is_lowercase() {
            return StrIndex::new(char_index, byte_index);
        }
    }

    if up {
        last_index
    } else {
        StrIndex::new(s.chars().count(), s.len())
    }
}

/// Returns index of the last camel-case component of `s`.
///
/// ```
/// assert_eq!(camel_case_start("AbcDef"), StrIndex::new(0, 0));
/// assert_eq!(camel_case_start("abcDef"), StrIndex::new(3, 3));
/// assert_eq!(camel_case_start("ABCD"), StrIndex::new(4, 4));
/// assert_eq!(camel_case_start("abcd"), StrIndex::new(4, 4));
/// assert_eq!(camel_case_start("\u{f6}\u{f6}cd"), StrIndex::new(4, 6));
/// ```
#[must_use]
pub fn camel_case_start(s: &str) -> StrIndex {
    let char_count = s.chars().count();
    let range = 0..char_count;
    let mut iter = range.rev().zip(s.char_indices().rev());
    if let Some((char_index, (_, first))) = iter.next() {
        if !first.is_lowercase() {
            return StrIndex::new(char_index, s.len());
        }
    } else {
        return StrIndex::new(char_count, s.len());
    }
    let mut down = true;
    let mut last_index = StrIndex::new(char_count, s.len());
    for (char_index, (byte_index, c)) in iter {
        if down {
            if c.is_uppercase() {
                down = false;
                last_index.byte_index = byte_index;
                last_index.char_index = char_index;
            } else if !c.is_lowercase() {
                return last_index;
            }
        } else if c.is_lowercase() {
            down = true;
        } else if c.is_uppercase() {
            last_index.byte_index = byte_index;
            last_index.char_index = char_index;
        } else {
            return last_index;
        }
    }
    last_index
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn camel_case_start_full() {
        assert_eq!(camel_case_start("AbcDef"), StrIndex::new(0, 0));
        assert_eq!(camel_case_start("Abc"), StrIndex::new(0, 0));
        assert_eq!(camel_case_start("ABcd"), StrIndex::new(0, 0));
        assert_eq!(camel_case_start("ABcdEf"), StrIndex::new(0, 0));
        assert_eq!(camel_case_start("AabABcd"), StrIndex::new(0, 0));
    }

    #[test]
    fn camel_case_start_partial() {
        assert_eq!(camel_case_start("abcDef"), StrIndex::new(3, 3));
        assert_eq!(camel_case_start("aDbc"), StrIndex::new(1, 1));
        assert_eq!(camel_case_start("aabABcd"), StrIndex::new(3, 3));
        assert_eq!(camel_case_start("\u{f6}\u{f6}AabABcd"), StrIndex::new(2, 4));
    }

    #[test]
    fn camel_case_start_not() {
        assert_eq!(camel_case_start("AbcDef_"), StrIndex::new(7, 7));
        assert_eq!(camel_case_start("AbcDD"), StrIndex::new(5, 5));
        assert_eq!(camel_case_start("all_small"), StrIndex::new(9, 9));
        assert_eq!(camel_case_start("\u{f6}_all_small"), StrIndex::new(11, 12));
    }

    #[test]
    fn camel_case_start_caps() {
        assert_eq!(camel_case_start("ABCD"), StrIndex::new(4, 4));
    }

    #[test]
    fn camel_case_until_full() {
        assert_eq!(camel_case_until("AbcDef"), StrIndex::new(6, 6));
        assert_eq!(camel_case_until("Abc"), StrIndex::new(3, 3));
        assert_eq!(camel_case_until("Abc\u{f6}\u{f6}\u{f6}"), StrIndex::new(6, 9));
    }

    #[test]
    fn camel_case_until_not() {
        assert_eq!(camel_case_until("abcDef"), StrIndex::new(0, 0));
        assert_eq!(camel_case_until("aDbc"), StrIndex::new(0, 0));
    }

    #[test]
    fn camel_case_until_partial() {
        assert_eq!(camel_case_until("AbcDef_"), StrIndex::new(6, 6));
        assert_eq!(camel_case_until("CallTypeC"), StrIndex::new(8, 8));
        assert_eq!(camel_case_until("AbcDD"), StrIndex::new(3, 3));
        assert_eq!(camel_case_until("Abc\u{f6}\u{f6}DD"), StrIndex::new(5, 7));
    }

    #[test]
    fn until_caps() {
        assert_eq!(camel_case_until("ABCD"), StrIndex::new(0, 0));
    }
}
