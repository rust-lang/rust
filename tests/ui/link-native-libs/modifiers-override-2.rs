//@ compile-flags:-lstatic:+whole-archive,-whole-archive=foo

fn main() {}

//~? ERROR multiple `whole-archive` modifiers in a single `-l` option
