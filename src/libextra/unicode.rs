// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[forbid(deprecated_mode)];
#[allow(missing_doc)];

pub mod icu {
    pub type UBool = u8;
    pub type UProperty = int;
    pub type UChar32 = char;

    pub static TRUE : u8 = 1u8;
    pub static FALSE : u8 = 1u8;

    pub static UCHAR_ALPHABETIC : UProperty = 0;
    pub static UCHAR_BINARY_START : UProperty = 0; // = UCHAR_ALPHABETIC
    pub static UCHAR_ASCII_HEX_DIGIT : UProperty = 1;
    pub static UCHAR_BIDI_CONTROL : UProperty = 2;

    pub static UCHAR_BIDI_MIRRORED : UProperty = 3;
    pub static UCHAR_DASH : UProperty = 4;
    pub static UCHAR_DEFAULT_IGNORABLE_CODE_POINT : UProperty = 5;
    pub static UCHAR_DEPRECATED : UProperty = 6;

    pub static UCHAR_DIACRITIC : UProperty = 7;
    pub static UCHAR_EXTENDER : UProperty = 8;
    pub static UCHAR_FULL_COMPOSITION_EXCLUSION : UProperty = 9;
    pub static UCHAR_GRAPHEME_BASE : UProperty = 10;

    pub static UCHAR_GRAPHEME_EXTEND : UProperty = 11;
    pub static UCHAR_GRAPHEME_LINK : UProperty = 12;
    pub static UCHAR_HEX_DIGIT : UProperty = 13;
    pub static UCHAR_HYPHEN : UProperty = 14;

    pub static UCHAR_ID_CONTINUE : UProperty = 15;
    pub static UCHAR_ID_START : UProperty = 16;
    pub static UCHAR_IDEOGRAPHIC : UProperty = 17;
    pub static UCHAR_IDS_BINARY_OPERATOR : UProperty = 18;

    pub static UCHAR_IDS_TRINARY_OPERATOR : UProperty = 19;
    pub static UCHAR_JOIN_CONTROL : UProperty = 20;
    pub static UCHAR_LOGICAL_ORDER_EXCEPTION : UProperty = 21;
    pub static UCHAR_LOWERCASE : UProperty = 22;

    pub static UCHAR_MATH : UProperty = 23;
    pub static UCHAR_NONCHARACTER_CODE_POINT : UProperty = 24;
    pub static UCHAR_QUOTATION_MARK : UProperty = 25;
    pub static UCHAR_RADICAL : UProperty = 26;

    pub static UCHAR_SOFT_DOTTED : UProperty = 27;
    pub static UCHAR_TERMINAL_PUNCTUATION : UProperty = 28;
    pub static UCHAR_UNIFIED_IDEOGRAPH : UProperty = 29;
    pub static UCHAR_UPPERCASE : UProperty = 30;

    pub static UCHAR_WHITE_SPACE : UProperty = 31;
    pub static UCHAR_XID_CONTINUE : UProperty = 32;
    pub static UCHAR_XID_START : UProperty = 33;
    pub static UCHAR_CASE_SENSITIVE : UProperty = 34;

    pub static UCHAR_S_TERM : UProperty = 35;
    pub static UCHAR_VARIATION_SELECTOR : UProperty = 36;
    pub static UCHAR_NFD_INERT : UProperty = 37;
    pub static UCHAR_NFKD_INERT : UProperty = 38;

    pub static UCHAR_NFC_INERT : UProperty = 39;
    pub static UCHAR_NFKC_INERT : UProperty = 40;
    pub static UCHAR_SEGMENT_STARTER : UProperty = 41;
    pub static UCHAR_PATTERN_SYNTAX : UProperty = 42;

    pub static UCHAR_PATTERN_WHITE_SPACE : UProperty = 43;
    pub static UCHAR_POSIX_ALNUM : UProperty = 44;
    pub static UCHAR_POSIX_BLANK : UProperty = 45;
    pub static UCHAR_POSIX_GRAPH : UProperty = 46;

    pub static UCHAR_POSIX_PRINT : UProperty = 47;
    pub static UCHAR_POSIX_XDIGIT : UProperty = 48;
    pub static UCHAR_CASED : UProperty = 49;
    pub static UCHAR_CASE_IGNORABLE : UProperty = 50;

    pub static UCHAR_CHANGES_WHEN_LOWERCASED : UProperty = 51;
    pub static UCHAR_CHANGES_WHEN_UPPERCASED : UProperty = 52;
    pub static UCHAR_CHANGES_WHEN_TITLECASED : UProperty = 53;
    pub static UCHAR_CHANGES_WHEN_CASEFOLDED : UProperty = 54;

    pub static UCHAR_CHANGES_WHEN_CASEMAPPED : UProperty = 55;
    pub static UCHAR_CHANGES_WHEN_NFKC_CASEFOLDED : UProperty = 56;
    pub static UCHAR_BINARY_LIMIT : UProperty = 57;
    pub static UCHAR_BIDI_CLASS : UProperty = 0x1000;

    pub static UCHAR_INT_START : UProperty = 0x1000; // UCHAR_BIDI_CLASS
    pub static UCHAR_BLOCK : UProperty = 0x1001;
    pub static UCHAR_CANONICAL_COMBINING_CLASS : UProperty = 0x1002;
    pub static UCHAR_DECOMPOSITION_TYPE : UProperty = 0x1003;

    pub static UCHAR_EAST_ASIAN_WIDTH : UProperty = 0x1004;
    pub static UCHAR_GENERAL_CATEGORY : UProperty = 0x1005;
    pub static UCHAR_JOINING_GROUP : UProperty = 0x1006;
    pub static UCHAR_JOINING_TYPE : UProperty = 0x1007;

    pub static UCHAR_LINE_BREAK : UProperty = 0x1008;
    pub static UCHAR_NUMERIC_TYPE : UProperty = 0x1009;
    pub static UCHAR_SCRIPT : UProperty = 0x100A;
    pub static UCHAR_HANGUL_SYLLABLE_TYPE : UProperty = 0x100B;

    pub static UCHAR_NFD_QUICK_CHECK : UProperty = 0x100C;
    pub static UCHAR_NFKD_QUICK_CHECK : UProperty = 0x100D;
    pub static UCHAR_NFC_QUICK_CHECK : UProperty = 0x100E;
    pub static UCHAR_NFKC_QUICK_CHECK : UProperty = 0x100F;

    pub static UCHAR_LEAD_CANONICAL_COMBINING_CLASS : UProperty = 0x1010;
    pub static UCHAR_TRAIL_CANONICAL_COMBINING_CLASS : UProperty = 0x1011;
    pub static UCHAR_GRAPHEME_CLUSTER_BREAK : UProperty = 0x1012;
    pub static UCHAR_SENTENCE_BREAK : UProperty = 0x1013;

    pub static UCHAR_WORD_BREAK : UProperty = 0x1014;
    pub static UCHAR_INT_LIMIT : UProperty = 0x1015;

    pub static UCHAR_GENERAL_CATEGORY_MASK : UProperty = 0x2000;
    pub static UCHAR_MASK_START : UProperty = 0x2000;
    // = UCHAR_GENERAL_CATEGORY_MASK
    pub static UCHAR_MASK_LIMIT : UProperty = 0x2001;

    pub static UCHAR_NUMERIC_VALUE : UProperty = 0x3000;
    pub static UCHAR_DOUBLE_START : UProperty = 0x3000;
    // = UCHAR_NUMERIC_VALUE
    pub static UCHAR_DOUBLE_LIMIT : UProperty = 0x3001;

    pub static UCHAR_AGE : UProperty = 0x4000;
    pub static UCHAR_STRING_START : UProperty = 0x4000; // = UCHAR_AGE
    pub static UCHAR_BIDI_MIRRORING_GLYPH : UProperty = 0x4001;
    pub static UCHAR_CASE_FOLDING : UProperty = 0x4002;

    pub static UCHAR_ISO_COMMENT : UProperty = 0x4003;
    pub static UCHAR_LOWERCASE_MAPPING : UProperty = 0x4004;
    pub static UCHAR_NAME : UProperty = 0x4005;
    pub static UCHAR_SIMPLE_CASE_FOLDING : UProperty = 0x4006;

    pub static UCHAR_SIMPLE_LOWERCASE_MAPPING : UProperty = 0x4007;
    pub static UCHAR_SIMPLE_TITLECASE_MAPPING : UProperty = 0x4008;
    pub static UCHAR_SIMPLE_UPPERCASE_MAPPING : UProperty = 0x4009;
    pub static UCHAR_TITLECASE_MAPPING : UProperty = 0x400A;

    pub static UCHAR_UNICODE_1_NAME : UProperty = 0x400B;
    pub static UCHAR_UPPERCASE_MAPPING : UProperty = 0x400C;
    pub static UCHAR_STRING_LIMIT : UProperty = 0x400D;

    pub static UCHAR_SCRIPT_EXTENSIONS : UProperty = 0x7000;
    pub static UCHAR_OTHER_PROPERTY_START : UProperty = 0x7000;
    // = UCHAR_SCRIPT_EXTENSIONS;
    pub static UCHAR_OTHER_PROPERTY_LIMIT : UProperty = 0x7001;

    pub static UCHAR_INVALID_CODE : UProperty = -1;

    pub mod libicu {
        #[link_name = "icuuc"]
        #[abi = "cdecl"]
        extern {
            pub unsafe fn u_hasBinaryProperty(c: UChar32, which: UProperty)
                                              -> UBool;
            pub unsafe fn u_isdigit(c: UChar32) -> UBool;
            pub unsafe fn u_islower(c: UChar32) -> UBool;
            pub unsafe fn u_isspace(c: UChar32) -> UBool;
            pub unsafe fn u_isupper(c: UChar32) -> UBool;
            pub unsafe fn u_tolower(c: UChar32) -> UChar32;
            pub unsafe fn u_toupper(c: UChar32) -> UChar32;
        }
    }
}

pub fn is_XID_start(c: char) -> bool {
    return icu::libicu::u_hasBinaryProperty(c, icu::UCHAR_XID_START)
        == icu::TRUE;
}

pub fn is_XID_continue(c: char) -> bool {
    return icu::libicu::u_hasBinaryProperty(c, icu::UCHAR_XID_START)
        == icu::TRUE;
}

/*
Function: is_digit

Returns true if a character is a digit.
*/
pub fn is_digit(c: char) -> bool {
    return icu::libicu::u_isdigit(c) == icu::TRUE;
}

/*
Function: is_lower

Returns true if a character is a lowercase letter.
*/
pub fn is_lower(c: char) -> bool {
    return icu::libicu::u_islower(c) == icu::TRUE;
}

/*
Function: is_space

Returns true if a character is space.
*/
pub fn is_space(c: char) -> bool {
    return icu::libicu::u_isspace(c) == icu::TRUE;
}

/*
Function: is_upper

Returns true if a character is an uppercase letter.
*/
pub fn is_upper(c: char) -> bool {
    return icu::libicu::u_isupper(c) == icu::TRUE;
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_is_digit() {
        assert!((unicode::icu::is_digit('0')));
        assert!((!unicode::icu::is_digit('m')));
    }

    #[test]
    fn test_is_lower() {
        assert!((unicode::icu::is_lower('m')));
        assert!((!unicode::icu::is_lower('M')));
    }

    #[test]
    fn test_is_space() {
        assert!((unicode::icu::is_space(' ')));
        assert!((!unicode::icu::is_space('m')));
    }

    #[test]
    fn test_is_upper() {
        assert!((unicode::icu::is_upper('M')));
        assert!((!unicode::icu::is_upper('m')));
    }
}
