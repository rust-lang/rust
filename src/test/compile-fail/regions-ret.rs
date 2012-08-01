fn f(_x : &a/int) -> &a/int {
    ret &3; //~ ERROR illegal borrow
}

fn main() {
}

