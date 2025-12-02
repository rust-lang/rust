//@aux-build:../../ui/auxiliary/proc_macro_unsafe.rs
//@revisions: default disabled
//@[default] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/undocumented_unsafe_blocks/default
//@[disabled] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/undocumented_unsafe_blocks/disabled

#![warn(clippy::undocumented_unsafe_blocks, clippy::unnecessary_safety_comment)]

mod unsafe_fns {
    /// SAFETY: Bla
    unsafe fn unsafe_doc_comment() {}
    //~^ unnecessary_safety_comment

    /**
     * SAFETY: Bla
     */
    unsafe fn unsafe_block_doc_comment() {}
    //~^ unnecessary_safety_comment
}

fn main() {}
