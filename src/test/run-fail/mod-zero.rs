// error-pattern:attempt to calculate the remainder with a divisor of zero

fn main() {
    let y = 0;
    let _z = 1 % y;
}
