#![deny(unreachable_patterns)]

fn main() {
    let foo = Some(1);
    match foo {
        Some(_) => {/* ... */}
        None => {/* ... */}
        _ => {/* ... */} //~ ERROR unreachable pattern
    }
}
