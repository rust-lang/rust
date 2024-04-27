fn main() {
    let (mut a, mut b);
    [a, .., b, ..] = [0, 1]; //~ ERROR `..` can only be used once per slice pattern
    [a, a, b] = [1, 2];
    //~^ ERROR pattern requires 3 elements but array has 2
    [_] = [1, 2];
    //~^ ERROR pattern requires 1 element but array has 2
}
