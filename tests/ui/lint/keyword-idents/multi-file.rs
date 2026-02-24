#![deny(keyword_idents)]
//@ edition: 2015

#[path = "./auxiliary/multi_file_submod.rs"]
mod multi_file_submod;
//~? ERROR `async` is a keyword
//~? WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2018!
//~? ERROR `await` is a keyword
//~? WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2018!
//~? ERROR `try` is a keyword
//~? WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2018!
//~? ERROR `dyn` is a keyword
//~? WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2018!
//~? ERROR `gen` is a keyword
//~? WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2024!

fn main() {}
