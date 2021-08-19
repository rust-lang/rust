extern crate m2 as next;

mod common_body;

pub fn linkage_chain() -> String {
    format!("m3:{} {}", crate::common_body::crate_type(), next::linkage_chain())
}
