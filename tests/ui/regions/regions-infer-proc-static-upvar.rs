// Test that, when a variable of type `&T` is captured inside a proc,
// we correctly infer/require that its lifetime is 'static.

fn foo<F:FnOnce()+'static>(_p: F) { }

static i: isize = 3;

fn capture_local() {
    let x = 3;
    let y = &x; //~ ERROR `x` does not live long enough
    foo(move|| {
        let _a = *y;
    });
}

fn capture_static() {
    // Legal because &i can have static lifetime:
    let y = &i;
    foo(move|| {
        let _a = *y;
    });
}

fn main() { }
