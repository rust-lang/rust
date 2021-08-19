mod common_body;

pub fn linkage_chain() -> String {
    format!("m1:{}", ::common_body::crate_type())
}
