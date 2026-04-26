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
fn test_case_mapping(
    ranges: &[(char, [char; 3])],
    lookup: fn(char) -> [char; 3],
    fallback: fn(char) -> [char; 3],
) {
    let mut start = '\u{80}';
    for &(key, val) in ranges {
        for c in start..key {
            assert_eq!(lookup(c), fallback(c), "{c:?}");
        }
        assert_eq!(lookup(key), val, "{key:?}");
        start = char::from_u32(key as u32 + 1).unwrap();
    }
    for c in start..=char::MAX {
        assert_eq!(lookup(c), fallback(c), "{c:?}");
    }
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn alphabetic() {
    test_boolean_property(test_data::ALPHABETIC, unicode_data::alphabetic::lookup);
    test_boolean_property(test_data::ALPHABETIC, char::is_alphabetic);
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn case_ignorable() {
    test_boolean_property(test_data::CASE_IGNORABLE, unicode_data::case_ignorable::lookup);
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn lt() {
    test_boolean_property(test_data::LT, unicode_data::lt::lookup);
    test_boolean_property(test_data::LT, char::is_titlecase);
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn grapheme_extend() {
    test_boolean_property(test_data::GRAPHEME_EXTEND, unicode_data::grapheme_extend::lookup);
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn lowercase() {
    test_boolean_property(test_data::LOWERCASE, unicode_data::lowercase::lookup);
    test_boolean_property(test_data::LOWERCASE, char::is_lowercase);
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn n() {
    test_boolean_property(test_data::N, unicode_data::n::lookup);
    test_boolean_property(test_data::N, char::is_numeric);
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn uppercase() {
    test_boolean_property(test_data::UPPERCASE, unicode_data::uppercase::lookup);
    test_boolean_property(test_data::UPPERCASE, char::is_uppercase);
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn white_space() {
    test_boolean_property(test_data::WHITE_SPACE, unicode_data::white_space::lookup);
    test_boolean_property(test_data::WHITE_SPACE, char::is_whitespace);
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn to_lowercase() {
    test_case_mapping(test_data::TO_LOWER, unicode_data::conversions::to_lower, |c| {
        [c, '\0', '\0']
    });
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn to_uppercase() {
    test_case_mapping(test_data::TO_UPPER, unicode_data::conversions::to_upper, |c| {
        [c, '\0', '\0']
    });
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn to_titlecase() {
    test_case_mapping(
        test_data::TO_TITLE,
        unicode_data::conversions::to_title,
        unicode_data::conversions::to_upper,
    );
}
