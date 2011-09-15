// error-pattern:can not return a reference to a mutable field

fn f(a: {mutable x: int}) -> &int {
    ret a.x;
}

fn main() {}
