//@edition:2021

fn foo() {
    const { return }
    //~^ ERROR: return statement outside of function body
}

fn labelled_block_break() {
    'a: { const { break 'a } }
    //~^ ERROR: `break` outside of a loop or labeled block
    //~| ERROR: use of unreachable label
}

fn loop_break() {
    loop {
        const { break }
        //~^ ERROR: `break` outside of a loop or labeled block
    }
}

fn continue_to_labelled_block() {
    'a: { const { continue 'a } }
    //~^ ERROR: `continue` outside of a loop
    //~| ERROR: use of unreachable label
}

fn loop_continue() {
    loop {
        const { continue }
        //~^ ERROR: `continue` outside of a loop
    }
}

async fn await_across_const_block() {
    const { async {}.await }
    //~^ ERROR: `await` is only allowed inside `async` functions and blocks
}

fn reference_to_non_constant_in_const_block() {
    let x = 1;
    const { &x };
    //~^ ERROR: attempt to use a non-constant value in a constant
}


fn main() {}
