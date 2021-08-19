extern crate m4 as next;

mod common_body;

pub fn linkage_chain() -> String {
    format!("m5:{} {}", crate::common_body::crate_type(), next::linkage_chain())
}
