fn foo(():(), ():()) {}
fn bar(():()) {}

struct S;
impl S {
    fn baz(self, (): ()) { }
    fn generic<T>(self, _: T) { }
}

fn main() {
    let _: Result<(), String> = Ok(); //~ ERROR this enum variant takes
    foo(); //~ ERROR function takes
    foo(()); //~ ERROR function takes
    bar(); //~ ERROR function takes
    S.baz(); //~ ERROR this method takes
    S.generic::<()>(); //~ ERROR this method takes
}
