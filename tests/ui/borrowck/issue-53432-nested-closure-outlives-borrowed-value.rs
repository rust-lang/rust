fn main() {
    let f = move || {};
    let _action = move || {
        || f() // The `nested` closure
        //~^ ERROR lifetime may not live long enough
    };
}
