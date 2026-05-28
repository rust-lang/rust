// Byte positions into inline assembly reported by codegen errors require normalization or else
// they may not identify the appropriate span.  Worse still, an ICE can occur if the erroneous
// span begins or ends part-way through a multibyte character.
//
// Regression test for https://github.com/rust-lang/rust/issues/110885

// This test is tied to assembler syntax and errors, which can vary by backend and architecture.
//@only-x86_64
//@needs-backends: llvm
//@build-fail

//~? ERROR instruction mnemonic
std::arch::global_asm!(include_str!("normalize-offsets-for-crlf.s"));
fn main() {}
