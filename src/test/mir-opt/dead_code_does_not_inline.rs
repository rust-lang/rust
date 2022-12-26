//EMIT_MIR dead_code_does_not_inline.main.Inline.after.mir
fn main() {
    if false {
        callee();
    }
}

fn callee() {
    println!("Wooooo I'm invisible");
}
