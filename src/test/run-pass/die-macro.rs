// Just testing that fail!() type checks in statement or expr

fn f() {
    fail!();

    let x: int = fail!();
}

pub fn main() {

}
