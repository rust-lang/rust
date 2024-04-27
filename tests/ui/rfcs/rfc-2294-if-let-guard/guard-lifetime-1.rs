// References to by-move bindings in an if-let guard *cannot* be used after the guard.

#![feature(if_let_guard)]

fn main() {
    let x: Option<Option<String>> = Some(Some(String::new()));
    match x {
        Some(mut y) if let Some(ref z) = y => {
            //~^ ERROR: cannot move out of `x.0` because it is borrowed
            let _z: &String = z;
            let _y: Option<String> = y;
        }
        _ => {}
    }
}
