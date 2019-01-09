// Test a method call where the parameter `B` would (illegally) be
// inferred to a region bound in the method argument. If this program
// were accepted, then the closure passed to `s.f` could escape its
// argument.

struct S;

impl S {
    fn f<B, F>(&self, _: F) where F: FnOnce(&i32) -> B {
    }
}

fn main() {
    let s = S;
    s.f(|p| p) //~ ERROR cannot infer
}
