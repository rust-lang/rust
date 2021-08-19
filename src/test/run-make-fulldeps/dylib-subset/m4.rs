extern crate m3 as next;

mod common_body;

pub fn linkage_chain() -> String {
    format!("m4:{} {}", ::common_body::crate_type(), next::linkage_chain())
}
