fn foo(():(), ():()) {}
fn bar(():()) {}

struct S;
impl S {
    fn baz(self, (): ()) { }
    fn generic<T>(self, _: T) { }
}

fn main() {
    let _: Result<(), String> = Ok(); //~ ERROR arguments to this function are incorrect
    foo(); //~ ERROR arguments to this function are incorrect
    foo(()); //~ ERROR arguments to this function are incorrect
    bar(); //~ ERROR arguments to this function are incorrect
    S.baz(); //~ ERROR arguments to this function are incorrect
    S.generic::<()>(); //~ ERROR arguments to this function are incorrect
}
