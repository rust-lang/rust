struct X<'a, T>(&'a T);

impl X<'_, _> {}
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for implementations

fn main() {}
