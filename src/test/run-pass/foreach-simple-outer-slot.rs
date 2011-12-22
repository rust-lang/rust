


// -*- rust -*-
fn main() {
    let sum: int = 0;
    first_ten {|i| #debug("main"); log_full(core::debug, i); sum = sum + i; };
    #debug("sum");
    log_full(core::debug, sum);
    assert (sum == 45);
}

fn first_ten(it: block(int)) {
    let i: int = 0;
    while i < 10 { #debug("first_ten"); it(i); i = i + 1; }
}
