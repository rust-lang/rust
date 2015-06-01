#![feature(plugin)]
#![plugin(clippy, regex_macros)]

extern crate regex;

#[test]
#[deny(mut_mut)]
fn test_regex() {
	let pattern = regex!(r"^(?P<level>[#]+)\s(?P<title>.+)$");
	assert!(pattern.is_match("# headline"));
}
