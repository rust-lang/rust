static S: str = todo!();
//~^ ERROR the size for values of type `str` cannot be known at compilation time

const C: str = todo!();
//~^ ERROR the size for values of type `str` cannot be known at compilation time

fn main() {}
