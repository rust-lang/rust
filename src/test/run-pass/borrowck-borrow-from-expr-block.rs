use ptr::to_uint;

fn borrow(x: &int, f: fn(x: &int)) {
    f(x)
}

fn test1(x: @~int) {
    // Right now, at least, this induces a copy of the unique pointer:
    do borrow({*x}) |p| {
        let x_a = ptr::addr_of(&(**x));
        assert (x_a as uint) != to_uint(p);
        assert unsafe{*x_a} == *p;
    }
}

fn main() {
    test1(@~22);
}