// A proc-macro that emits a `core::arch::global_asm!` block whose template
// depends on `PROC_MACRO_ASM_TOKEN`, read at expansion time. The source of
// this crate is stable across the test; only the env var differs between
// runs, so the only thing that changes in the consumer is the asm template
// spliced in by the macro.
//
// The body is a pure assembler comment, so it assembles to nothing on every
// target we test on.

extern crate proc_macro;

use proc_macro::TokenStream;

#[proc_macro]
pub fn emit_global_asm(_input: TokenStream) -> TokenStream {
    let value = std::env::var("PROC_MACRO_ASM_TOKEN").unwrap();
    format!(r##"core::arch::global_asm!("# {}");"##, value).parse().unwrap()
}
