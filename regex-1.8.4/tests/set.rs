matset!(set1, &["a", "a"], "a", 0, 1);
matset!(set2, &["a", "a"], "ba", 0, 1);
matset!(set3, &["a", "b"], "a", 0);
matset!(set4, &["a", "b"], "b", 1);
matset!(set5, &["a|b", "b|a"], "b", 0, 1);
matset!(set6, &["foo", "oo"], "foo", 0, 1);
matset!(set7, &["^foo", "bar$"], "foo", 0);
matset!(set8, &["^foo", "bar$"], "foo bar", 0, 1);
matset!(set9, &["^foo", "bar$"], "bar", 1);
matset!(set10, &[r"[a-z]+$", "foo"], "01234 foo", 0, 1);
matset!(set11, &[r"[a-z]+$", "foo"], "foo 01234", 1);
matset!(set12, &[r".*?", "a"], "zzzzzza", 0, 1);
matset!(set13, &[r".*", "a"], "zzzzzza", 0, 1);
matset!(set14, &[r".*", "a"], "zzzzzz", 0);
matset!(set15, &[r"(?-u)\ba\b"], "hello a bye", 0);
matset!(set16, &["a"], "a", 0);
matset!(set17, &[".*a"], "a", 0);
matset!(set18, &["a", "β"], "β", 1);

// regexes that match the empty string
matset!(setempty1, &["", "a"], "abc", 0, 1);
matset!(setempty2, &["", "b"], "abc", 0, 1);
matset!(setempty3, &["", "z"], "abc", 0);
matset!(setempty4, &["a", ""], "abc", 0, 1);
matset!(setempty5, &["b", ""], "abc", 0, 1);
matset!(setempty6, &["z", ""], "abc", 1);
matset!(setempty7, &["b", "(?:)"], "abc", 0, 1);
matset!(setempty8, &["(?:)", "b"], "abc", 0, 1);
matset!(setempty9, &["c(?:)", "b"], "abc", 0, 1);

nomatset!(nset1, &["a", "a"], "b");
nomatset!(nset2, &["^foo", "bar$"], "bar foo");
nomatset!(
    nset3,
    {
        let xs: &[&str] = &[];
        xs
    },
    "a"
);
nomatset!(nset4, &[r"^rooted$", r"\.log$"], "notrooted");

// See: https://github.com/rust-lang/regex/issues/187
#[test]
fn regression_subsequent_matches() {
    let set = regex_set!(&["ab", "b"]);
    let text = text!("ba");
    assert!(set.matches(text).matched(1));
    assert!(set.matches(text).matched(1));
}

#[test]
fn get_set_patterns() {
    let set = regex_set!(&["a", "b"]);
    assert_eq!(vec!["a", "b"], set.patterns());
}

#[test]
fn len_and_empty() {
    let empty = regex_set!(&[""; 0]);
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());

    let not_empty = regex_set!(&["ab", "b"]);
    assert_eq!(not_empty.len(), 2);
    assert!(!not_empty.is_empty());
}

#[test]
fn default_set_is_empty() {
    let set: regex::bytes::RegexSet = Default::default();
    assert_eq!(set.len(), 0);
    assert!(set.is_empty());
}
