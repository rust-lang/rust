// Regression test for https://github.com/rust-lang/rust/issues/152831

fn if_let_binding() {
    if let Some(v) = Some(1) {
        const { v }
        //~^ ERROR: attempt to use a non-constant value in a constant
    }
}

fn while_let_binding() {
    while let Some(v) = Some(1) {
        const { v }
        //~^ ERROR: attempt to use a non-constant value in a constant
        break;
    }
}

fn let_else_binding() {
    let Some(v) = Some(1) else { return };
    const { v }
    //~^ ERROR: attempt to use a non-constant value in a constant
}

fn main() {}
