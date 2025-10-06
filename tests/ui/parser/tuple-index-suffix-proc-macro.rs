//! See #59418.
//!
//! Like `tuple-index-suffix.rs`, but exercises the proc-macro interaction.

//@ proc-macro: tuple-index-suffix-proc-macro-aux.rs
//@ ignore-backends: gcc

extern crate tuple_index_suffix_proc_macro_aux;
use tuple_index_suffix_proc_macro_aux as aux;

fn main() {
    struct TupStruct(i32);
    let tup_struct = TupStruct(42);

    // Previously, #60186 had carve outs for `{i,u}{32,usize}` as non-lint pseudo-FCW warnings. Now,
    // they all hard error.

    aux::bad_tup_indexing!(0usize);
    //~^ ERROR suffixes on a tuple index are invalid
    aux::bad_tup_struct_indexing!(tup_struct, 0isize);
    //~^ ERROR suffixes on a tuple index are invalid

    // Not part of the #60186 carve outs.

    aux::bad_tup_indexing!(0u8);
    //~^ ERROR suffixes on a tuple index are invalid
    aux::bad_tup_struct_indexing!(tup_struct, 0u64);
    //~^ ERROR suffixes on a tuple index are invalid

    // NOTE: didn't bother with trying to figure out how to generate `struct P { 0u32: u32 }` using
    // *only* `proc_macro` without help with `syn`/`quote`, looks like you can't with just
    // `proc_macro::quote`?
}
