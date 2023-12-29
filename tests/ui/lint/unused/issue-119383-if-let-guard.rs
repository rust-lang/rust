#![feature(if_let_guard)]
#![deny(unused_variables)]

fn main() {
    match () {
        () if let Some(b) = Some(()) => {} //~ ERROR unused variable: `b`
        _ => {}
    }
}
