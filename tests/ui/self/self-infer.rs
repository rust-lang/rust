struct S;

impl S {
    fn f(self: _) {} //~ERROR the placeholder `_` is not allowed within types on item signatures for methods
    fn g(self: &_) {} //~ERROR the placeholder `_` is not allowed within types on item signatures for methods
}

fn main() {}
