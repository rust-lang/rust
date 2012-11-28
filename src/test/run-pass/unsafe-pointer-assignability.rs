fn f(x: *int) {
    unsafe {
        assert *x == 3;
    }
}

fn main() {
    f(&3);
}



