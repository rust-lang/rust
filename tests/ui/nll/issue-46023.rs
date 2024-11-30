//@ check-pass
fn main() {
    let x = 0;

    (move || {
        x = 1;
        //~^ WARNING cannot assign to `x`, as it is not declared as mutable [E0594]
    })()
}
