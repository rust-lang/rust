fn foo(():(), ():()) {}
fn bar(():()) {}

struct S;
impl S {
    fn baz(self, (): ()) { }
    fn generic<T>(self, _: T) { }
}

fn main() {
    let _: Result<(), String> = Ok(); //~ ERROR this function takes
    foo(); //~ ERROR this function takes
    foo(()); //~ ERROR this function takes
    bar(); //~ ERROR this function takes
    S.baz(); //~ ERROR this function takes
    S.generic::<()>(); //~ ERROR this function takes
}
