// References to by-mutable-ref bindings in an if-let guard *can* be used after the guard.

//@ check-pass

#![feature(if_let_guard)]

fn main() {
    let mut x: Option<Option<String>> = Some(Some(String::new()));
    match x {
        Some(ref mut y) if let Some(ref z) = *y => {
            let _z: &String = z;
            let _y: &mut Option<String> = y;
        }
        _ => {}
    }
}
