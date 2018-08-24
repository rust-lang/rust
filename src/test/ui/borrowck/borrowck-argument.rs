#[derive(Copy, Clone)]
struct S;

impl S {
    fn mutate(&mut self) {
    }
}

fn func(arg: S) {
    arg.mutate(); //~ ERROR: cannot borrow immutable argument
}

impl S {
    fn method(&self, arg: S) {
        arg.mutate(); //~ ERROR: cannot borrow immutable argument
    }
}

trait T {
    fn default(&self, arg: S) {
        arg.mutate(); //~ ERROR: cannot borrow immutable argument
    }
}

impl T for S {}

fn main() {
    let s = S;
    func(s);
    s.method(s);
    s.default(s);
    (|arg: S| { arg.mutate() })(s); //~ ERROR: cannot borrow immutable argument
}
