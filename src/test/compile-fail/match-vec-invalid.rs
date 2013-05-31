fn main() {
    let a = ~[];
    match a {
        [1, ..tail, ..tail] => {}, //~ ERROR: unexpected token: `..`
        _ => ()
    }
}
