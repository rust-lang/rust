// Issue #2040

fn main() {
    let foo = 1;
    assert ptr::addr_of(foo) == ptr::addr_of(foo);
}
