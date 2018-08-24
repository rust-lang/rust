struct S;

impl S {
    fn f(self: _) {} //~ERROR the type placeholder `_` is not allowed within types on item sig
    fn g(self: &_) {} //~ERROR the type placeholder `_` is not allowed within types on item sig
}

fn main() {}
