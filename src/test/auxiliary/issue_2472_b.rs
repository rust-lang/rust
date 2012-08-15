
enum S = ();

impl S {
    fn foo() { }
}

trait T {
    fn bar();
}

impl S: T {
    fn bar() { }
}
