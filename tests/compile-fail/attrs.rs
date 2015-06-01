#![feature(plugin)]
#![plugin(clippy)]

#[deny(inline_always)]
#[inline(always)] //~ERROR You have declared #[inline(always)] on test_attr_lint.
fn test_attr_lint() {
	assert!(true)
}

fn main() {
	test_attr_lint()
}
