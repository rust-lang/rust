use core::unicode::unicode_data;
use std::ops::RangeInclusive;

mod test_data;

#[test]
pub fn version() {
    let (major, _minor, _update) = core::char::UNICODE_VERSION;
    assert!(major >= 10);
}

#[track_caller]
fn test_boolean_property(ranges: &[RangeInclusive<char>], lookup: fn(char) -> bool) {
    let mut start = '\u{80}';
    for range in ranges {
        for c in start..*range.start() {
            assert!(!lookup(c), "{c:?}");
        }
        for c in range.clone() {
            assert!(lookup(c), "{c:?}");
        }
        start = char::from_u32(*range.end() as u32 + 1).unwrap();
    }
    for c in start..=char::MAX {
        assert!(!lookup(c), "{c:?}");
    }
}

#[track_caller]
fn test_case_mapping(ranges: &[(char, [char; 3])], lookup: fn(char) -> [char; 3]) {
    let mut start = '\u{80}';
    for &(key, val) in ranges {
        for c in start..key {
            assert_eq!(lookup(c), [c, '\0', '\0'], "{c:?}");
        }
        assert_eq!(lookup(key), val, "{key:?}");
        start = char::from_u32(key as u32 + 1).unwrap();
    }
    for c in start..=char::MAX {
        assert_eq!(lookup(c), [c, '\0', '\0'], "{c:?}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn alphabetic() {
    test_boolean_property(test_data::ALPHABETIC, unicode_data::alphabetic::lookup);
}

#[test]
#[cfg_attr(miri, ignore)]
fn case_ignorable() {
    test_boolean_property(test_data::CASE_IGNORABLE, unicode_data::case_ignorable::lookup);
}

#[test]
#[cfg_attr(miri, ignore)]
fn cased() {
    test_boolean_property(test_data::CASED, unicode_data::cased::lookup);
}

#[test]
#[cfg_attr(miri, ignore)]
fn grapheme_extend() {
    test_boolean_property(test_data::GRAPHEME_EXTEND, unicode_data::grapheme_extend::lookup);
}

#[test]
#[cfg_attr(miri, ignore)]
fn lowercase() {
    test_boolean_property(test_data::LOWERCASE, unicode_data::lowercase::lookup);
}

#[test]
fn n() {
    test_boolean_property(test_data::N, unicode_data::n::lookup);
}

#[test]
#[cfg_attr(miri, ignore)]
fn uppercase() {
    test_boolean_property(test_data::UPPERCASE, unicode_data::uppercase::lookup);
}

#[test]
#[cfg_attr(miri, ignore)]
fn white_space() {
    test_boolean_property(test_data::WHITE_SPACE, unicode_data::white_space::lookup);
}

#[test]
#[cfg_attr(miri, ignore)]
fn to_lowercase() {
    test_case_mapping(test_data::TO_LOWER, unicode_data::conversions::to_lower);
}

#[test]
#[cfg_attr(miri, ignore)]
fn to_uppercase() {
    test_case_mapping(test_data::TO_UPPER, unicode_data::conversions::to_upper);
}
