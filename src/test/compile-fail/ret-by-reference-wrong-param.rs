// error-pattern:can not return a reference to the wrong parameter

fn f(a: int, b: int) -> &2 int {
    ret a;
}

fn main() {}
