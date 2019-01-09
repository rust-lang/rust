fn main() {
    let a = Vec::new();
    match a {
        [1, tail.., tail..] => {}, //~ ERROR: expected one of `,` or `@`, found `..`
        _ => ()
    }
}
