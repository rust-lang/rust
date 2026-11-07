//@ check-pass

fn main() {
    let c = 1;
    let w = "T";
    match Some(5) {
        None if c == 1 && (w != "Y" && w != "E") => {}
        _ => panic!(),
    }
}
