fn main() {
    let Some(x) = Some(1) else { //~ ERROR does not diverge
        Some(2)
    };
    let Some(x) = Some(1) else { //~ ERROR does not diverge
        if 1 == 1 {
            panic!();
        }
    };
    let Some(x) = Some(1) else { Some(2) }; //~ ERROR does not diverge
}
