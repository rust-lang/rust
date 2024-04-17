macro_rules! shortmat {
    ($name:ident, $re:expr, $text:expr, $shortest_match:expr) => {
        #[test]
        fn $name() {
            let text = text!($text);
            let re = regex!($re);
            assert_eq!($shortest_match, re.shortest_match(text));
        }
    };
}

shortmat!(t01, r"a+", r"aa", Some(1));
// Test that the reverse suffix optimization gets it right.
shortmat!(t02, r".*(?:abcd)+", r"abcdabcd", Some(4));
