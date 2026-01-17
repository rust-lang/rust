#[macro_use]
extern crate proc_macro_lib;

#[derive(PanicDerive)]
pub struct PanicStruct;

#[panic_attr]
mod my_mod {}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("derive") => PanicStruct::do_panic(),
        Some("attr") => generated_panic(),
        _ => println!("usage: main [derive|attr]"),
    }
}
