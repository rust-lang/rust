// Test that `assert` works in consts.

const _: () = assert!(true);

const _: () = assert!(false);
//~^ ERROR evaluation of constant value failed

fn main() {}
