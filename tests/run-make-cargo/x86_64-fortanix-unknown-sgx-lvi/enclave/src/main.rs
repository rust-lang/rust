std::arch::global_asm!(
    r#"
    .text
    .global rust_plus_one_global_asm
    .type rust_plus_one_global_asm, @function
rust_plus_one_global_asm:
    movl (%rdi), %eax
    inc %eax
    retq
"#,
    options(att_syntax)
);

extern "C" {
    fn cc_plus_one_c(arg: &u32) -> u32;
    fn cc_plus_one_c_asm(arg: &u32) -> u32;
    fn cc_plus_one_cxx(arg: &u32) -> u32;
    fn cc_plus_one_cxx_asm(arg: &u32) -> u32;
    fn cc_plus_one_asm(arg: &u32) -> u32;
    fn cmake_plus_one_c(arg: &u32) -> u32;
    fn cmake_plus_one_c_asm(arg: &u32) -> u32;
    fn cmake_plus_one_cxx(arg: &u32) -> u32;
    fn cmake_plus_one_cxx_asm(arg: &u32) -> u32;
    fn cmake_plus_one_c_global_asm(arg: &u32) -> u32;
    fn cmake_plus_one_cxx_global_asm(arg: &u32) -> u32;
    fn cmake_plus_one_asm(arg: &u32) -> u32;
    fn rust_plus_one_global_asm(arg: &u32) -> u32;
}

fn main() {
    let value: u32 = 41;
    let question = "Answer to the Ultimate Question of Life, the Universe, and Everything:";

    unsafe {
        println!("{}: {}!", question, rust_plus_one_global_asm(&value));
        println!("{}: {}!", question, cc_plus_one_c(&value));
        println!("{}: {}!", question, cc_plus_one_c_asm(&value));
        println!("{}: {}!", question, cc_plus_one_cxx(&value));
        println!("{}: {}!", question, cc_plus_one_cxx_asm(&value));
        println!("{}: {}!", question, cc_plus_one_asm(&value));
        println!("{}: {}!", question, cmake_plus_one_c(&value));
        println!("{}: {}!", question, cmake_plus_one_c_asm(&value));
        println!("{}: {}!", question, cmake_plus_one_cxx(&value));
        println!("{}: {}!", question, cmake_plus_one_cxx_asm(&value));
        println!("{}: {}!", question, cmake_plus_one_c_global_asm(&value));
        println!("{}: {}!", question, cmake_plus_one_cxx_global_asm(&value));
        println!("{}: {}!", question, cmake_plus_one_asm(&value));
    }
}
