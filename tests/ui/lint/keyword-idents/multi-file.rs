#![deny(keyword_idents)] // Should affect the submodule
//@ edition: 2015
//~? ERROR: `async` is a keyword in the 2018 edition
//~? WARN this is accepted in the current edition
//~? ERROR: `await` is a keyword in the 2018 edition
//~? WARN this is accepted in the current edition
//~? ERROR: `dyn` is a keyword in the 2018 edition
//~? WARN this is accepted in the current edition
//~? ERROR: `try` is a keyword in the 2018 edition
//~? WARN this is accepted in the current edition
//~? ERROR: `gen` is a keyword in the 2024 edition
//~? WARN this is accepted in the current edition


#[path = "./auxiliary/multi_file_submod.rs"]
mod multi_file_submod;

fn main() {}
