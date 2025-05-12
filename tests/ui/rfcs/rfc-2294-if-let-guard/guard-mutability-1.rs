// Check mutable bindings cannot be mutated by an if-let guard.

#![feature(if_let_guard)]

fn main() {
    let x: Option<Option<i32>> = Some(Some(6));
    match x {
        Some(mut y) if let Some(ref mut z) = y => {
            //~^ ERROR cannot borrow `y.0` as mutable, as it is immutable for the pattern guard
            let _: &mut i32 = z;
        }
        _ => {}
    }
}
