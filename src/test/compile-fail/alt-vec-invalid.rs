fn main() {
    let a = ~[];
    match a {
        [1, ..tail, ..tail] => {}, //~ ERROR: expected `]` but found `,`
        _ => ()
    }
}
