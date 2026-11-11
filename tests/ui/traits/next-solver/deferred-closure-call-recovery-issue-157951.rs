//@ compile-flags: -Znext-solver=globally
//@ check-fail

fn main() {
    let f = |f: dyn Fn()| f;
    //~^ ERROR the size for values of type `(dyn Fn() + 'static)` cannot be known at compilation time
    //~| ERROR return type cannot be a trait object without pointer indirection
    f();
    //~^ ERROR this function takes 1 argument but 0 arguments were supplied
}
