// compile-flags: -Zunstable-options
// edition: 2024
#![feature(new_temp_lifetime)]

fn foo() {}

fn id<T>(t: T) -> T { t }

fn f<T>(_: &T) {}

fn main() {
    let x = id({
        super let y = 1;
        &y //~ ERROR: `y` does not live long enough
    }); // ... because `y` is freed here
    let x = id(&foo()); //~ ERROR: temporary value dropped while borrowed
        // lifetime of y is equal to
        // what lifetime of block *would* be if it were
        // an `&`-rvalue expression (e.g., `&{...}`)
    f(x);
}
