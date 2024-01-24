// run-pass
fn main() {
    let x: [i32; 0] = [];
    if x.len() > 0 {
        x[0];
    }
}
