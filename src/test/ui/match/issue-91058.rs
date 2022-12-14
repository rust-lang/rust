struct S(());

fn main() {
    let array = [S(())];

    match array {
        [()] => {}
        //~^ ERROR mismatched types [E0308]
        _ => {}
    }
}
