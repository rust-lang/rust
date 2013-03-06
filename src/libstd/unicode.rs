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

pub mod icu {
    pub type UBool = u8;
    pub type UProperty = int;
    pub type UChar32 = char;

    pub const TRUE : u8 = 1u8;
    pub const FALSE : u8 = 1u8;

    pub const UCHAR_ALPHABETIC : UProperty = 0;
    pub const UCHAR_BINARY_START : UProperty = 0; // = UCHAR_ALPHABETIC
    pub const UCHAR_ASCII_HEX_DIGIT : UProperty = 1;
    pub const UCHAR_BIDI_CONTROL : UProperty = 2;

    pub const UCHAR_BIDI_MIRRORED : UProperty = 3;
    pub const UCHAR_DASH : UProperty = 4;
    pub const UCHAR_DEFAULT_IGNORABLE_CODE_POINT : UProperty = 5;
    pub const UCHAR_DEPRECATED : UProperty = 6;

    pub const UCHAR_DIACRITIC : UProperty = 7;
    pub const UCHAR_EXTENDER : UProperty = 8;
    pub const UCHAR_FULL_COMPOSITION_EXCLUSION : UProperty = 9;
    pub const UCHAR_GRAPHEME_BASE : UProperty = 10;

    pub const UCHAR_GRAPHEME_EXTEND : UProperty = 11;
    pub const UCHAR_GRAPHEME_LINK : UProperty = 12;
    pub const UCHAR_HEX_DIGIT : UProperty = 13;
    pub const UCHAR_HYPHEN : UProperty = 14;

    pub const UCHAR_ID_CONTINUE : UProperty = 15;
    pub const UCHAR_ID_START : UProperty = 16;
    pub const UCHAR_IDEOGRAPHIC : UProperty = 17;
    pub const UCHAR_IDS_BINARY_OPERATOR : UProperty = 18;

    pub const UCHAR_IDS_TRINARY_OPERATOR : UProperty = 19;
    pub const UCHAR_JOIN_CONTROL : UProperty = 20;
    pub const UCHAR_LOGICAL_ORDER_EXCEPTION : UProperty = 21;
    pub const UCHAR_LOWERCASE : UProperty = 22;

    pub const UCHAR_MATH : UProperty = 23;
    pub const UCHAR_NONCHARACTER_CODE_POINT : UProperty = 24;
    pub const UCHAR_QUOTATION_MARK : UProperty = 25;
    pub const UCHAR_RADICAL : UProperty = 26;

    pub const UCHAR_SOFT_DOTTED : UProperty = 27;
    pub const UCHAR_TERMINAL_PUNCTUATION : UProperty = 28;
    pub const UCHAR_UNIFIED_IDEOGRAPH : UProperty = 29;
    pub const UCHAR_UPPERCASE : UProperty = 30;

    pub const UCHAR_WHITE_SPACE : UProperty = 31;
    pub const UCHAR_XID_CONTINUE : UProperty = 32;
    pub const UCHAR_XID_START : UProperty = 33;
    pub const UCHAR_CASE_SENSITIVE : UProperty = 34;

    pub const UCHAR_S_TERM : UProperty = 35;
    pub const UCHAR_VARIATION_SELECTOR : UProperty = 36;
    pub const UCHAR_NFD_INERT : UProperty = 37;
    pub const UCHAR_NFKD_INERT : UProperty = 38;

    pub const UCHAR_NFC_INERT : UProperty = 39;
    pub const UCHAR_NFKC_INERT : UProperty = 40;
    pub const UCHAR_SEGMENT_STARTER : UProperty = 41;
    pub const UCHAR_PATTERN_SYNTAX : UProperty = 42;

    pub const UCHAR_PATTERN_WHITE_SPACE : UProperty = 43;
    pub const UCHAR_POSIX_ALNUM : UProperty = 44;
    pub const UCHAR_POSIX_BLANK : UProperty = 45;
    pub const UCHAR_POSIX_GRAPH : UProperty = 46;

    pub const UCHAR_POSIX_PRINT : UProperty = 47;
    pub const UCHAR_POSIX_XDIGIT : UProperty = 48;
    pub const UCHAR_CASED : UProperty = 49;
    pub const UCHAR_CASE_IGNORABLE : UProperty = 50;

    pub const UCHAR_CHANGES_WHEN_LOWERCASED : UProperty = 51;
    pub const UCHAR_CHANGES_WHEN_UPPERCASED : UProperty = 52;
    pub const UCHAR_CHANGES_WHEN_TITLECASED : UProperty = 53;
    pub const UCHAR_CHANGES_WHEN_CASEFOLDED : UProperty = 54;

    pub const UCHAR_CHANGES_WHEN_CASEMAPPED : UProperty = 55;
    pub const UCHAR_CHANGES_WHEN_NFKC_CASEFOLDED : UProperty = 56;
    pub const UCHAR_BINARY_LIMIT : UProperty = 57;
    pub const UCHAR_BIDI_CLASS : UProperty = 0x1000;

    pub const UCHAR_INT_START : UProperty = 0x1000; // UCHAR_BIDI_CLASS
    pub const UCHAR_BLOCK : UProperty = 0x1001;
    pub const UCHAR_CANONICAL_COMBINING_CLASS : UProperty = 0x1002;
    pub const UCHAR_DECOMPOSITION_TYPE : UProperty = 0x1003;

    pub const UCHAR_EAST_ASIAN_WIDTH : UProperty = 0x1004;
    pub const UCHAR_GENERAL_CATEGORY : UProperty = 0x1005;
    pub const UCHAR_JOINING_GROUP : UProperty = 0x1006;
    pub const UCHAR_JOINING_TYPE : UProperty = 0x1007;

    pub const UCHAR_LINE_BREAK : UProperty = 0x1008;
    pub const UCHAR_NUMERIC_TYPE : UProperty = 0x1009;
    pub const UCHAR_SCRIPT : UProperty = 0x100A;
    pub const UCHAR_HANGUL_SYLLABLE_TYPE : UProperty = 0x100B;

    pub const UCHAR_NFD_QUICK_CHECK : UProperty = 0x100C;
    pub const UCHAR_NFKD_QUICK_CHECK : UProperty = 0x100D;
    pub const UCHAR_NFC_QUICK_CHECK : UProperty = 0x100E;
    pub const UCHAR_NFKC_QUICK_CHECK : UProperty = 0x100F;

    pub const UCHAR_LEAD_CANONICAL_COMBINING_CLASS : UProperty = 0x1010;
    pub const UCHAR_TRAIL_CANONICAL_COMBINING_CLASS : UProperty = 0x1011;
    pub const UCHAR_GRAPHEME_CLUSTER_BREAK : UProperty = 0x1012;
    pub const UCHAR_SENTENCE_BREAK : UProperty = 0x1013;

    pub const UCHAR_WORD_BREAK : UProperty = 0x1014;
    pub const UCHAR_INT_LIMIT : UProperty = 0x1015;

    pub const UCHAR_GENERAL_CATEGORY_MASK : UProperty = 0x2000;
    pub const UCHAR_MASK_START : UProperty = 0x2000;
    // = UCHAR_GENERAL_CATEGORY_MASK
    pub const UCHAR_MASK_LIMIT : UProperty = 0x2001;

    pub const UCHAR_NUMERIC_VALUE : UProperty = 0x3000;
    pub const UCHAR_DOUBLE_START : UProperty = 0x3000;
    // = UCHAR_NUMERIC_VALUE
    pub const UCHAR_DOUBLE_LIMIT : UProperty = 0x3001;

    pub const UCHAR_AGE : UProperty = 0x4000;
    pub const UCHAR_STRING_START : UProperty = 0x4000; // = UCHAR_AGE
    pub const UCHAR_BIDI_MIRRORING_GLYPH : UProperty = 0x4001;
    pub const UCHAR_CASE_FOLDING : UProperty = 0x4002;

    pub const UCHAR_ISO_COMMENT : UProperty = 0x4003;
    pub const UCHAR_LOWERCASE_MAPPING : UProperty = 0x4004;
    pub const UCHAR_NAME : UProperty = 0x4005;
    pub const UCHAR_SIMPLE_CASE_FOLDING : UProperty = 0x4006;

    pub const UCHAR_SIMPLE_LOWERCASE_MAPPING : UProperty = 0x4007;
    pub const UCHAR_SIMPLE_TITLECASE_MAPPING : UProperty = 0x4008;
    pub const UCHAR_SIMPLE_UPPERCASE_MAPPING : UProperty = 0x4009;
    pub const UCHAR_TITLECASE_MAPPING : UProperty = 0x400A;

    pub const UCHAR_UNICODE_1_NAME : UProperty = 0x400B;
    pub const UCHAR_UPPERCASE_MAPPING : UProperty = 0x400C;
    pub const UCHAR_STRING_LIMIT : UProperty = 0x400D;

    pub const UCHAR_SCRIPT_EXTENSIONS : UProperty = 0x7000;
    pub const UCHAR_OTHER_PROPERTY_START : UProperty = 0x7000;
    // = UCHAR_SCRIPT_EXTENSIONS;
    pub const UCHAR_OTHER_PROPERTY_LIMIT : UProperty = 0x7001;

    pub const UCHAR_INVALID_CODE : UProperty = -1;

    pub mod libicu {
        #[link_name = "icuuc"]
        #[abi = "cdecl"]
        pub extern {
            unsafe fn u_hasBinaryProperty(c: UChar32, which: UProperty)
                                       -> UBool;
            unsafe fn u_isdigit(c: UChar32) -> UBool;
            unsafe fn u_islower(c: UChar32) -> UBool;
            unsafe fn u_isspace(c: UChar32) -> UBool;
            unsafe fn u_isupper(c: UChar32) -> UBool;
            unsafe fn u_tolower(c: UChar32) -> UChar32;
            unsafe fn u_toupper(c: UChar32) -> UChar32;
        }
    }
}

pub pure fn is_XID_start(c: char) -> bool {
    return icu::libicu::u_hasBinaryProperty(c, icu::UCHAR_XID_START)
        == icu::TRUE;
}

pub pure fn is_XID_continue(c: char) -> bool {
    return icu::libicu::u_hasBinaryProperty(c, icu::UCHAR_XID_START)
        == icu::TRUE;
}

/*
Function: is_digit

Returns true if a character is a digit.
*/
pub pure fn is_digit(c: char) -> bool {
    return icu::libicu::u_isdigit(c) == icu::TRUE;
}

/*
Function: is_lower

Returns true if a character is a lowercase letter.
*/
pub pure fn is_lower(c: char) -> bool {
    return icu::libicu::u_islower(c) == icu::TRUE;
}

/*
Function: is_space

Returns true if a character is space.
*/
pub pure fn is_space(c: char) -> bool {
    return icu::libicu::u_isspace(c) == icu::TRUE;
}

/*
Function: is_upper

Returns true if a character is an uppercase letter.
*/
pub pure fn is_upper(c: char) -> bool {
    return icu::libicu::u_isupper(c) == icu::TRUE;
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_is_digit() {
        fail_unless!((unicode::icu::is_digit('0')));
        fail_unless!((!unicode::icu::is_digit('m')));
    }

    #[test]
    fn test_is_lower() {
        fail_unless!((unicode::icu::is_lower('m')));
        fail_unless!((!unicode::icu::is_lower('M')));
    }

    #[test]
    fn test_is_space() {
        fail_unless!((unicode::icu::is_space(' ')));
        fail_unless!((!unicode::icu::is_space('m')));
    }

    #[test]
    fn test_is_upper() {
        fail_unless!((unicode::icu::is_upper('M')));
        fail_unless!((!unicode::icu::is_upper('m')));
    }
}
