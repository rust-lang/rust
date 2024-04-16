//@ known-bug: #121623
fn main() {
    match () {
        _ => 'b: {
            continue 'b;
        }
    }
}
