// error-pattern:can not return a reference to a function-local value

fn f(a: {mutable x: int}) -> &int {
    let x = {y: 4};
    ret x.y;
}

fn main() {}
