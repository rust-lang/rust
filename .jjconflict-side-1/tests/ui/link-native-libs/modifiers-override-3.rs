// Regression test for issue #97299, one command line library with modifiers
// overrides another command line library with modifiers.

//@ compile-flags:-lstatic:+whole-archive=foo -lstatic:+whole-archive=foo

fn main() {}

//~? ERROR overriding linking modifiers from command line is not supported
