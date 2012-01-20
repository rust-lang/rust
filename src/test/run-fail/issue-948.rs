// error-pattern:beep boop
fn main() {
    let origin = {x: 0, y: 0};
    let f = {x: (fail "beep boop") with origin};
}
