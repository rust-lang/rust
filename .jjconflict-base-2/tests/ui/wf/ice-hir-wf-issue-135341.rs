type A<T> = B;
type B = _; //~ ERROR the placeholder `_` is not allowed within types on item signatures for type aliases

fn main() {}
