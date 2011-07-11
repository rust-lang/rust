// error-pattern:did not expect a record with a field q

fn main() {
    alt rec(x=1, y=2) {
        {x, q} {}
    }
}
