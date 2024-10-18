//@ run-rustfix

fn f() {}
fn i(_: u32) {}
fn is(_: u32, _: &str) {}
fn s(_: &str) {}

fn main() {
    // code             expected suggestion
    f(0, 1,);        // f()
    //~^ error: this function takes 0 arguments but 2 arguments were supplied
    i(0, 1, 2,);     // i(0,)
    //~^ error: this function takes 1 argument but 3 arguments were supplied
    i(0, 1, 2);      // i(0)
    //~^ error: this function takes 1 argument but 3 arguments were supplied
    is(0, 1, 2, ""); // is(0, "")
    //~^ error: this function takes 2 arguments but 4 arguments were supplied
    is((), 1, "", ());
    //~^ error: this function takes 2 arguments but 4 arguments were supplied
    is(1, (), "", ());
    //~^ error: this function takes 2 arguments but 4 arguments were supplied
    s(0, 1, "");     // s("")
    //~^ error: this function takes 1 argument but 3 arguments were supplied
}
