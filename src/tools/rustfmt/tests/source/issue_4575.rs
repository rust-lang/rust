//@ rustfmt-hard_tabs: true

fn main() {
    let condition_a = true;
    let condition_b = false;
    let x = 123.456789
        + if condition_a {
            x + y + z + w
        } else {
            123.456789
        }
        + if condition_b {
            x - y - z - w
        } else {
            123.456789
        };
}
