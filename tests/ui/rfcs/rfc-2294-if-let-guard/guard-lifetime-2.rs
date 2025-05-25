// References to by-mutable-ref bindings in an if-let guard *can* be used after the guard.

//@ check-pass
//@revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024

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
