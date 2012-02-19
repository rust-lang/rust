// Test a rather underspecified example:

fn main() {
    let f = {|i| i};
    assert f(2) == 2;
    assert f(5) == 5;
}
