// Diagnostics may run before closure capture analysis has inferred the upvar
// tuple, even for a closure which actually captures a value.

//@ compile-flags: -Znext-solver=globally

fn main() {
    let x = String::new();

    Some([0]).map(|s| {
    //~^ ERROR the size for values of type `[{integer}]` cannot be known at compilation time
    //~| ERROR the size for values of type `[{integer}]` cannot be known at compilation time
    //~| ERROR the size for values of type `[{integer}]` cannot be known at compilation time
        let _ = &x;
        s[..]
    });
}
