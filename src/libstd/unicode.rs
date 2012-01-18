
mod icu {
    type UBool = u8;
    type UProperty = int;
    type UChar32 = char;

    const TRUE : u8 = 1u8;
    const FALSE : u8 = 1u8;

    const UCHAR_ALPHABETIC : UProperty = 0;
    const UCHAR_BINARY_START : UProperty = 0; // = UCHAR_ALPHABETIC
    const UCHAR_ASCII_HEX_DIGIT : UProperty = 1;
    const UCHAR_BIDI_CONTROL : UProperty = 2;

    const UCHAR_BIDI_MIRRORED : UProperty = 3;
    const UCHAR_DASH : UProperty = 4;
    const UCHAR_DEFAULT_IGNORABLE_CODE_POINT : UProperty = 5;
    const UCHAR_DEPRECATED : UProperty = 6;

    const UCHAR_DIACRITIC : UProperty = 7;
    const UCHAR_EXTENDER : UProperty = 8;
    const UCHAR_FULL_COMPOSITION_EXCLUSION : UProperty = 9;
    const UCHAR_GRAPHEME_BASE : UProperty = 10;

    const UCHAR_GRAPHEME_EXTEND : UProperty = 11;
    const UCHAR_GRAPHEME_LINK : UProperty = 12;
    const UCHAR_HEX_DIGIT : UProperty = 13;
    const UCHAR_HYPHEN : UProperty = 14;

    const UCHAR_ID_CONTINUE : UProperty = 15;
    const UCHAR_ID_START : UProperty = 16;
    const UCHAR_IDEOGRAPHIC : UProperty = 17;
    const UCHAR_IDS_BINARY_OPERATOR : UProperty = 18;

    const UCHAR_IDS_TRINARY_OPERATOR : UProperty = 19;
    const UCHAR_JOIN_CONTROL : UProperty = 20;
    const UCHAR_LOGICAL_ORDER_EXCEPTION : UProperty = 21;
    const UCHAR_LOWERCASE : UProperty = 22;

    const UCHAR_MATH : UProperty = 23;
    const UCHAR_NONCHARACTER_CODE_POINT : UProperty = 24;
    const UCHAR_QUOTATION_MARK : UProperty = 25;
    const UCHAR_RADICAL : UProperty = 26;

    const UCHAR_SOFT_DOTTED : UProperty = 27;
    const UCHAR_TERMINAL_PUNCTUATION : UProperty = 28;
    const UCHAR_UNIFIED_IDEOGRAPH : UProperty = 29;
    const UCHAR_UPPERCASE : UProperty = 30;

    const UCHAR_WHITE_SPACE : UProperty = 31;
    const UCHAR_XID_CONTINUE : UProperty = 32;
    const UCHAR_XID_START : UProperty = 33;
    const UCHAR_CASE_SENSITIVE : UProperty = 34;

    const UCHAR_S_TERM : UProperty = 35;
    const UCHAR_VARIATION_SELECTOR : UProperty = 36;
    const UCHAR_NFD_INERT : UProperty = 37;
    const UCHAR_NFKD_INERT : UProperty = 38;

    const UCHAR_NFC_INERT : UProperty = 39;
    const UCHAR_NFKC_INERT : UProperty = 40;
    const UCHAR_SEGMENT_STARTER : UProperty = 41;
    const UCHAR_PATTERN_SYNTAX : UProperty = 42;

    const UCHAR_PATTERN_WHITE_SPACE : UProperty = 43;
    const UCHAR_POSIX_ALNUM : UProperty = 44;
    const UCHAR_POSIX_BLANK : UProperty = 45;
    const UCHAR_POSIX_GRAPH : UProperty = 46;

    const UCHAR_POSIX_PRINT : UProperty = 47;
    const UCHAR_POSIX_XDIGIT : UProperty = 48;
    const UCHAR_CASED : UProperty = 49;
    const UCHAR_CASE_IGNORABLE : UProperty = 50;

    const UCHAR_CHANGES_WHEN_LOWERCASED : UProperty = 51;
    const UCHAR_CHANGES_WHEN_UPPERCASED : UProperty = 52;
    const UCHAR_CHANGES_WHEN_TITLECASED : UProperty = 53;
    const UCHAR_CHANGES_WHEN_CASEFOLDED : UProperty = 54;

    const UCHAR_CHANGES_WHEN_CASEMAPPED : UProperty = 55;
    const UCHAR_CHANGES_WHEN_NFKC_CASEFOLDED : UProperty = 56;
    const UCHAR_BINARY_LIMIT : UProperty = 57;
    const UCHAR_BIDI_CLASS : UProperty = 0x1000;

    const UCHAR_INT_START : UProperty = 0x1000; // UCHAR_BIDI_CLASS
    const UCHAR_BLOCK : UProperty = 0x1001;
    const UCHAR_CANONICAL_COMBINING_CLASS : UProperty = 0x1002;
    const UCHAR_DECOMPOSITION_TYPE : UProperty = 0x1003;

    const UCHAR_EAST_ASIAN_WIDTH : UProperty = 0x1004;
    const UCHAR_GENERAL_CATEGORY : UProperty = 0x1005;
    const UCHAR_JOINING_GROUP : UProperty = 0x1006;
    const UCHAR_JOINING_TYPE : UProperty = 0x1007;

    const UCHAR_LINE_BREAK : UProperty = 0x1008;
    const UCHAR_NUMERIC_TYPE : UProperty = 0x1009;
    const UCHAR_SCRIPT : UProperty = 0x100A;
    const UCHAR_HANGUL_SYLLABLE_TYPE : UProperty = 0x100B;

    const UCHAR_NFD_QUICK_CHECK : UProperty = 0x100C;
    const UCHAR_NFKD_QUICK_CHECK : UProperty = 0x100D;
    const UCHAR_NFC_QUICK_CHECK : UProperty = 0x100E;
    const UCHAR_NFKC_QUICK_CHECK : UProperty = 0x100F;

    const UCHAR_LEAD_CANONICAL_COMBINING_CLASS : UProperty = 0x1010;
    const UCHAR_TRAIL_CANONICAL_COMBINING_CLASS : UProperty = 0x1011;
    const UCHAR_GRAPHEME_CLUSTER_BREAK : UProperty = 0x1012;
    const UCHAR_SENTENCE_BREAK : UProperty = 0x1013;

    const UCHAR_WORD_BREAK : UProperty = 0x1014;
    const UCHAR_INT_LIMIT : UProperty = 0x1015;

    const UCHAR_GENERAL_CATEGORY_MASK : UProperty = 0x2000;
    const UCHAR_MASK_START : UProperty = 0x2000;
    // = UCHAR_GENERAL_CATEGORY_MASK
    const UCHAR_MASK_LIMIT : UProperty = 0x2001;

    const UCHAR_NUMERIC_VALUE : UProperty = 0x3000;
    const UCHAR_DOUBLE_START : UProperty = 0x3000;
    // = UCHAR_NUMERIC_VALUE
    const UCHAR_DOUBLE_LIMIT : UProperty = 0x3001;

    const UCHAR_AGE : UProperty = 0x4000;
    const UCHAR_STRING_START : UProperty = 0x4000; // = UCHAR_AGE
    const UCHAR_BIDI_MIRRORING_GLYPH : UProperty = 0x4001;
    const UCHAR_CASE_FOLDING : UProperty = 0x4002;

    const UCHAR_ISO_COMMENT : UProperty = 0x4003;
    const UCHAR_LOWERCASE_MAPPING : UProperty = 0x4004;
    const UCHAR_NAME : UProperty = 0x4005;
    const UCHAR_SIMPLE_CASE_FOLDING : UProperty = 0x4006;

    const UCHAR_SIMPLE_LOWERCASE_MAPPING : UProperty = 0x4007;
    const UCHAR_SIMPLE_TITLECASE_MAPPING : UProperty = 0x4008;
    const UCHAR_SIMPLE_UPPERCASE_MAPPING : UProperty = 0x4009;
    const UCHAR_TITLECASE_MAPPING : UProperty = 0x400A;

    const UCHAR_UNICODE_1_NAME : UProperty = 0x400B;
    const UCHAR_UPPERCASE_MAPPING : UProperty = 0x400C;
    const UCHAR_STRING_LIMIT : UProperty = 0x400D;

    const UCHAR_SCRIPT_EXTENSIONS : UProperty = 0x7000;
    const UCHAR_OTHER_PROPERTY_START : UProperty = 0x7000;
    // = UCHAR_SCRIPT_EXTENSIONS;
    const UCHAR_OTHER_PROPERTY_LIMIT : UProperty = 0x7001;

    const UCHAR_INVALID_CODE : UProperty = 0xffffffff;
    // FIXME: should be -1, change when compiler supports negative
    // constants

    #[link_name = "icuuc"]
    #[abi = "cdecl"]
    native mod libicu {
        pure fn u_hasBinaryProperty(c: UChar32, which: UProperty) -> UBool;
        pure fn u_isdigit(c: UChar32) -> UBool;
        pure fn u_islower(c: UChar32) -> UBool;
        pure fn u_isspace(c: UChar32) -> UBool;
        pure fn u_isupper(c: UChar32) -> UBool;
        pure fn u_tolower(c: UChar32) -> UChar32;
        pure fn u_toupper(c: UChar32) -> UChar32;
    }
}

pure fn is_XID_start(c: char) -> bool {
    ret icu::libicu::u_hasBinaryProperty(c, icu::UCHAR_XID_START)
        == icu::TRUE;
}

pure fn is_XID_continue(c: char) -> bool {
    ret icu::libicu::u_hasBinaryProperty(c, icu::UCHAR_XID_START)
        == icu::TRUE;
}

/*
Function: is_digit

Returns true if a character is a digit.
*/
pure fn is_digit(c: char) -> bool {
    ret icu::libicu::u_isdigit(c) == icu::TRUE;
}

/*
Function: is_lower

Returns true if a character is a lowercase letter.
*/
pure fn is_lower(c: char) -> bool {
    ret icu::libicu::u_islower(c) == icu::TRUE;
}

/*
Function: is_space

Returns true if a character is space.
*/
pure fn is_space(c: char) -> bool {
    ret icu::libicu::u_isspace(c) == icu::TRUE;
}

/*
Function: is_upper

Returns true if a character is an uppercase letter.
*/
pure fn is_upper(c: char) -> bool {
    ret icu::libicu::u_isupper(c) == icu::TRUE;
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_is_digit() {
        assert (unicode::icu::is_digit('0'));
        assert (!unicode::icu::is_digit('m'));
    }

    #[test]
    fn test_is_lower() {
        assert (unicode::icu::is_lower('m'));
        assert (!unicode::icu::is_lower('M'));
    }

    #[test]
    fn test_is_space() {
        assert (unicode::icu::is_space(' '));
        assert (!unicode::icu::is_space('m'));
    }

    #[test]
    fn test_is_upper() {
        assert (unicode::icu::is_upper('M'));
        assert (!unicode::icu::is_upper('m'));
    }
}