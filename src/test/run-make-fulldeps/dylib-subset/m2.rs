extern crate m1 as next;

mod common_body;

pub fn linkage_chain() -> String {
    format!("m2:{} {}", crate::common_body::crate_type(), next::linkage_chain())
}
