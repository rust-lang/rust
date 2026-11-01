// Regression test for issue #5239

fn main() {
    let x = |ref x: isize| { x += 1; };
    //~^ ERROR E0368
}
