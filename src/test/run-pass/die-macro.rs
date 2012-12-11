// Just testing that die!() type checks in statement or expr

fn f() {
    die!();

    let x: int = die!();
}

fn main() {

}
