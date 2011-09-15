// error-pattern:can not return a reference to a temporary

fn f(a: int) -> &int {
    ret 10;
}

fn main() {}
