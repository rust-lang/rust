//@ run-pass
fn mir() -> (){
    let x = 1;
    let mut y = 0;
    while  y < x {
        y += 1
    }
}

pub fn main() {
    mir();
}
