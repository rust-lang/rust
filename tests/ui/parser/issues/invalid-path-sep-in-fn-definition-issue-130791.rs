//@ run-rustfix

#[allow(dead_code)]
fn invalid_path_separator::<T>() {}
//~^ ERROR invalid path separator in function definition

fn main() {}
