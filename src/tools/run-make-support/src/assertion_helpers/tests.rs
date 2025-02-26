//! Basic sanity checks for assertion helpers.
use super::*;

mod test_assert_equals {
    use super::*;

    #[test]
    fn assert_equals_same() {
        assert_equals("foo", "foo");
        assert_equals("", "");
    }

    #[test]
    #[should_panic]
    fn assert_equals_different() {
        assert_equals("foo", "bar");
    }
}

mod test_assert_contains {
    use super::*;

    #[test]
    fn assert_contains_yes() {
        assert_contains("", "");
        assert_contains(" ", "");
        assert_contains("a", "a");
        assert_contains("ab", "a");
    }

    #[test]
    #[should_panic]
    fn assert_contains_no() {
        assert_contains("a", "b");
    }
}

mod test_assert_not_contains {
    use super::*;

    #[test]
    fn assert_not_contains_yes() {
        assert_not_contains("a", "b");
    }

    #[test]
    #[should_panic]
    fn assert_not_contains_no() {
        assert_not_contains(" ", "");
    }
}

mod assert_contains_regex {
    use super::*;

    #[test]
    fn assert_contains_regex_yes() {
        assert_contains_regex("", "");
        assert_contains_regex("", ".*");
        assert_contains_regex("abcde", ".*");
        assert_contains_regex("abcde", ".+");
    }

    #[test]
    #[should_panic]
    fn assert_contains_regex_no() {
        assert_contains_regex("", ".+");
    }
}

mod assert_not_contains_regex_regex {
    use super::*;

    #[test]
    fn assert_not_contains_regex_yes() {
        assert_not_contains_regex("abc", "d");
    }

    #[test]
    #[should_panic]
    fn assert_not_contains_regex_no() {
        assert_not_contains_regex("abc", ".*");
    }
}

mod test_assert_count_is {
    use super::*;

    #[test]
    fn assert_count_is_yes() {
        assert_count_is(0, "", "b");
        assert_count_is(3, "abcbdb", "b");
    }

    #[test]
    #[should_panic]
    fn assert_count_is_no() {
        assert_count_is(2, "abcbdb", "b");
    }
}
