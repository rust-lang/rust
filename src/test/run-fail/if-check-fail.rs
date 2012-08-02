// error-pattern:Number is odd
pure fn even(x: uint) -> bool {
    if x < 2u {
        return false;
    } else if x == 2u { return true; } else { return even(x - 2u); }
}

fn foo(x: uint) {
    if even(x) {
        log(debug, x);
    } else {
        fail ~"Number is odd";
    }
}

fn main() { foo(3u); }
