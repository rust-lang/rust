fn main() {}

const Z: () = panic!("cheese");
//~^ ERROR panicking in constants is unstable

const Y: () = unreachable!();
//~^ ERROR panicking in constants is unstable

const X: () = unimplemented!();
//~^ ERROR panicking in constants is unstable
