// xfail-fast  (compile-flags unsupported on windows)
// compile-flags:--borrowck=err
// exec-env:RUST_POISON_ON_FREE=1

fn borrow(x: &int, f: fn(x: &int)) {
    let before = *x;
    f(x);
    let after = *x;
    assert before == after;
}

fn main() {
    let mut x = @3;
    borrow(x) {|b_x|
        assert *b_x == 3;
        assert ptr::addr_of(*x) == ptr::addr_of(*b_x);
        x = @22;

        #debug["ptr::addr_of(*b_x) = %x", ptr::addr_of(*b_x) as uint];
        assert *b_x == 3;
        assert ptr::addr_of(*x) != ptr::addr_of(*b_x);
    }
}