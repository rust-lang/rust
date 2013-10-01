// Just testing that fail!() type checks in statement or expr

#[allow(unreachable_code)];

fn f() {
    fail2!();

    let _x: int = fail2!();
}

pub fn main() {

}
