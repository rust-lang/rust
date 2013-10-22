// Just testing that fail!() type checks in statement or expr

#[allow(unreachable_code)];

fn f() {
    fail!();

    let _x: int = fail!();
}

pub fn main() {

}
