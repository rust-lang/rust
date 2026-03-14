// Ensure that non-constant values in pattern bindings (e.g. `if let`, `while let`,
// `let ... else`) do not produce incorrect suggestions to use `const` instead of `let`.

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
