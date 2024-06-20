//@ run-pass
// When all branches of an if expression result in panic, the entire if
// expression results in panic.

pub fn main() {
    let _x = if true {
        10
    } else {
        if true { panic!() } else { panic!() }
    };
}
