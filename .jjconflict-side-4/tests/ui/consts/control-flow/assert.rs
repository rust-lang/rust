// Test that `assert` works in consts.

const _: () = assert!(true);

const _: () = assert!(false);
//~^ ERROR assertion failed

fn main() {}
