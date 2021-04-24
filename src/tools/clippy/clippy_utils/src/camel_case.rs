/// Returns the index of the character after the first camel-case component of `s`.
#[must_use]
pub fn until(s: &str) -> usize {
    let mut iter = s.char_indices();
    if let Some((_, first)) = iter.next() {
        if !first.is_uppercase() {
            return 0;
        }
    } else {
        return 0;
    }
    let mut up = true;
    let mut last_i = 0;
    for (i, c) in iter {
        if up {
            if c.is_lowercase() {
                up = false;
            } else {
                return last_i;
            }
        } else if c.is_uppercase() {
            up = true;
            last_i = i;
        } else if !c.is_lowercase() {
            return i;
        }
    }
    if up { last_i } else { s.len() }
}

/// Returns index of the last camel-case component of `s`.
#[must_use]
pub fn from(s: &str) -> usize {
    let mut iter = s.char_indices().rev();
    if let Some((_, first)) = iter.next() {
        if !first.is_lowercase() {
            return s.len();
        }
    } else {
        return s.len();
    }
    let mut down = true;
    let mut last_i = s.len();
    for (i, c) in iter {
        if down {
            if c.is_uppercase() {
                down = false;
                last_i = i;
            } else if !c.is_lowercase() {
                return last_i;
            }
        } else if c.is_lowercase() {
            down = true;
        } else if c.is_uppercase() {
            last_i = i;
        } else {
            return last_i;
        }
    }
    last_i
}

#[cfg(test)]
mod test {
    use super::{from, until};

    #[test]
    fn from_full() {
        assert_eq!(from("AbcDef"), 0);
        assert_eq!(from("Abc"), 0);
        assert_eq!(from("ABcd"), 0);
        assert_eq!(from("ABcdEf"), 0);
        assert_eq!(from("AabABcd"), 0);
    }

    #[test]
    fn from_partial() {
        assert_eq!(from("abcDef"), 3);
        assert_eq!(from("aDbc"), 1);
        assert_eq!(from("aabABcd"), 3);
    }

    #[test]
    fn from_not() {
        assert_eq!(from("AbcDef_"), 7);
        assert_eq!(from("AbcDD"), 5);
    }

    #[test]
    fn from_caps() {
        assert_eq!(from("ABCD"), 4);
    }

    #[test]
    fn until_full() {
        assert_eq!(until("AbcDef"), 6);
        assert_eq!(until("Abc"), 3);
    }

    #[test]
    fn until_not() {
        assert_eq!(until("abcDef"), 0);
        assert_eq!(until("aDbc"), 0);
    }

    #[test]
    fn until_partial() {
        assert_eq!(until("AbcDef_"), 6);
        assert_eq!(until("CallTypeC"), 8);
        assert_eq!(until("AbcDD"), 3);
    }

    #[test]
    fn until_caps() {
        assert_eq!(until("ABCD"), 0);
    }
}
