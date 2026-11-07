// Suggestion to use impl trait in closure parameter is invalid, see issue 138932
fn main() {
    let c = |f: dyn Fn()| f();
    //~^ ERROR: the size for values of type `(dyn Fn() + 'static)` cannot be known at compilation time
}
