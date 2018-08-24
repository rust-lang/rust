// error-pattern:Number is odd
fn even(x: usize) -> bool {
    if x < 2 {
        return false;
    } else if x == 2 {
        return true;
    } else {
        return even(x - 2);
    }
}

fn foo(x: usize) {
    if even(x) {
        println!("{}", x);
    } else {
        panic!("Number is odd");
    }
}

fn main() {
    foo(3);
}
