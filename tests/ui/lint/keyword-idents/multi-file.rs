#![deny(keyword_idents)] // Should affect the submodule.
//@ edition: 2015

// `keyword_idents_2018` and `keyword_idents_2024` are pre-expansion lints,
// and configuring them via lint attributes has to propagate to submodules
// in other files.
// <https://github.com/rust-lang/rust/issues/132218>

#[path = "./auxiliary/multi_file_submod.rs"]
mod multi_file_submod;

fn main() {}

//~? ERROR `async` is a keyword in the 2018 edition
//~? WARN this is accepted in the current edition
//~? ERROR `await` is a keyword in the 2018 edition
//~? WARN this is accepted in the current edition
//~? ERROR `try` is a keyword in the 2018 edition
//~? WARN this is accepted in the current edition
//~? ERROR `dyn` is a keyword in the 2018 edition
//~? WARN this is accepted in the current edition
//~? ERROR `gen` is a keyword in the 2024 edition
//~? WARN this is accepted in the current edition
