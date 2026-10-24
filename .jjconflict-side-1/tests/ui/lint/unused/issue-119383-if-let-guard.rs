#![deny(unused_variables)]

fn main() {
    match () {
        () if let Some(b) = Some(()) => {} //~ ERROR unused variable: `b`
        _ => {}
    }
}
