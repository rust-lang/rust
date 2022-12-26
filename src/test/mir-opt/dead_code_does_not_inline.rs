// EMIT_MIR dead_code_does_not_inline.main.Inline.after.mir
pub fn main() {
    if false {
        callee();
    }
}

pub fn callee() {
    println!("Wooooo I'm invisible");
}
