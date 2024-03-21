macro_rules! noparse(
    ($name:ident, $re:expr) => (
        #[test]
        fn $name() {
            let re = $re;
            match regex_new!(re) {
                Err(_) => {},
                Ok(_) => panic!("Regex '{}' should cause a parse error.", re),
            }
        }
    );
);

noparse!(fail_no_repeat_arg, "*");
noparse!(fail_incomplete_escape, "\\");
noparse!(fail_class_incomplete, "[A-");
noparse!(fail_class_not_closed, "[A");
noparse!(fail_class_no_begin, r"[\A]");
noparse!(fail_class_no_end, r"[\z]");
noparse!(fail_class_no_boundary, r"[\b]");
noparse!(fail_open_paren, "(");
noparse!(fail_close_paren, ")");
noparse!(fail_invalid_range, "[a-Z]");
noparse!(fail_empty_capture_name, "(?P<>a)");
noparse!(fail_bad_capture_name, "(?P<na-me>)");
noparse!(fail_bad_flag, "(?a)a");
noparse!(fail_too_big, "a{10000000}");
noparse!(fail_counted_no_close, "a{1001");
noparse!(fail_counted_decreasing, "a{2,1}");
noparse!(fail_counted_nonnegative, "a{-1,1}");
noparse!(fail_unfinished_cap, "(?");
noparse!(fail_unfinished_escape, "\\");
noparse!(fail_octal_digit, r"\8");
noparse!(fail_hex_digit, r"\xG0");
noparse!(fail_hex_short, r"\xF");
noparse!(fail_hex_long_digits, r"\x{fffg}");
noparse!(fail_flag_bad, "(?a)");
noparse!(fail_flag_empty, "(?)");
noparse!(fail_double_neg, "(?-i-i)");
noparse!(fail_neg_empty, "(?i-)");
noparse!(fail_dupe_named, "(?P<a>.)(?P<a>.)");
noparse!(fail_range_end_no_class, "[a-[:lower:]]");
noparse!(fail_range_end_no_begin, r"[a-\A]");
noparse!(fail_range_end_no_end, r"[a-\z]");
noparse!(fail_range_end_no_boundary, r"[a-\b]");
