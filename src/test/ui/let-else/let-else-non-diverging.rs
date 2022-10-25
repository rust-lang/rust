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

    // Ensure that uninhabited types do not "diverge".
    // This might be relaxed in the future, but when it is,
    // it should be an explicitly wanted decision.
    let Some(x) = Some(1) else { foo::<Uninhabited>() }; //~ ERROR does not diverge
}

enum Uninhabited {}

fn foo<T>() -> T {
    panic!()
}
