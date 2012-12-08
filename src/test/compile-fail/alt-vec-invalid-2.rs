fn main() {
    match ~[] {
        [_, ..tail, _] => {}, //~ ERROR: expected `]` but found `,`
        _ => ()
    }
}
