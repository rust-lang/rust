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

/// Dealing with sting comparison can be complicated, this struct ensures that both the
/// character and byte count are provided for correct indexing.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct StrCount {
    pub char_count: usize,
    pub byte_count: usize,
}

impl StrCount {
    pub fn new(char_count: usize, byte_count: usize) -> Self {
        Self { char_count, byte_count }
    }
}

/// Returns the number of chars that match from the start
///
/// ```
/// assert_eq!(count_match_start("hello_mouse", "hello_penguin"), StrCount::new(6, 6));
/// assert_eq!(count_match_start("hello_clippy", "bye_bugs"), StrCount::new(0, 0));
/// assert_eq!(count_match_start("hello_world", "hello_world"), StrCount::new(11, 11));
/// assert_eq!(count_match_start("T\u{f6}ffT\u{f6}ff", "T\u{f6}ff"), StrCount::new(4, 5));
/// ```
#[must_use]
pub fn count_match_start(str1: &str, str2: &str) -> StrCount {
    // (char_index, char1)
    let char_count = str1.chars().count();
    let iter1 = (0..=char_count).zip(str1.chars());
    // (byte_index, char2)
    let iter2 = str2.char_indices();

    iter1
        .zip(iter2)
        .take_while(|((_, c1), (_, c2))| c1 == c2)
        .last()
        .map_or_else(StrCount::default, |((char_index, _), (byte_index, character))| {
            StrCount::new(char_index + 1, byte_index + character.len_utf8())
        })
}

/// Returns the number of chars and bytes that match from the end
///
/// ```
/// assert_eq!(count_match_end("hello_cat", "bye_cat"), StrCount::new(4, 4));
/// assert_eq!(count_match_end("if_item_thing", "enum_value"), StrCount::new(0, 0));
/// assert_eq!(count_match_end("Clippy", "Clippy"), StrCount::new(6, 6));
/// assert_eq!(count_match_end("MyT\u{f6}ff", "YourT\u{f6}ff"), StrCount::new(4, 5));
/// ```
#[must_use]
pub fn count_match_end(str1: &str, str2: &str) -> StrCount {
    let char_count = str1.chars().count();
    if char_count == 0 {
        return StrCount::default();
    }

    // (char_index, char1)
    let iter1 = (0..char_count).rev().zip(str1.chars().rev());
    // (byte_index, char2)
    let byte_count = str2.len();
    let iter2 = str2.char_indices().rev();

    iter1
        .zip(iter2)
        .take_while(|((_, c1), (_, c2))| c1 == c2)
        .last()
        .map_or_else(StrCount::default, |((char_index, _), (byte_index, _))| {
            StrCount::new(char_count - char_index, byte_count - byte_index)
        })
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
