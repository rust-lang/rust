/// Return the index of the character after the first camel-case component of
/// `s`.
pub fn camel_case_until(s: &str) -> usize {
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
    if up {
        last_i
    } else {
        s.len()
    }
}

/// Return index of the last camel-case component of `s`.
pub fn camel_case_from(s: &str) -> usize {
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
        } else {
            return last_i;
        }
    }
    last_i
}

#[cfg(test)]
mod test {
    use super::{camel_case_from, camel_case_until};

    #[test]
    fn from_full() {
        assert_eq!(camel_case_from("AbcDef"), 0);
        assert_eq!(camel_case_from("Abc"), 0);
    }

    #[test]
    fn from_partial() {
        assert_eq!(camel_case_from("abcDef"), 3);
        assert_eq!(camel_case_from("aDbc"), 1);
    }

    #[test]
    fn from_not() {
        assert_eq!(camel_case_from("AbcDef_"), 7);
        assert_eq!(camel_case_from("AbcDD"), 5);
    }

    #[test]
    fn from_caps() {
        assert_eq!(camel_case_from("ABCD"), 4);
    }

    #[test]
    fn until_full() {
        assert_eq!(camel_case_until("AbcDef"), 6);
        assert_eq!(camel_case_until("Abc"), 3);
    }

    #[test]
    fn until_not() {
        assert_eq!(camel_case_until("abcDef"), 0);
        assert_eq!(camel_case_until("aDbc"), 0);
    }

    #[test]
    fn until_partial() {
        assert_eq!(camel_case_until("AbcDef_"), 6);
        assert_eq!(camel_case_until("CallTypeC"), 8);
        assert_eq!(camel_case_until("AbcDD"), 3);
    }

    #[test]
    fn until_caps() {
        assert_eq!(camel_case_until("ABCD"), 0);
    }
}