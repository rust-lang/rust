// issue-119209

fn main<'a>(_: &'a i32) -> &'a () { &() } //~ERROR `main` function return type is not allowed to have generic parameters
