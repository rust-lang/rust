extern crate m5 as next;

mod common_body;

pub fn linkage_chain() -> String {
    format!("m6:{} {}", ::common_body::crate_type(), next::linkage_chain())
}

fn main() {
    println!("linkage: {}", linkage_chain());
}
