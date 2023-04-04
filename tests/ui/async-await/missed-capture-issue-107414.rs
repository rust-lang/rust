// check-pass
// edition:2018

fn main() {}

struct StructA {}
struct StructB {}

impl StructA {
    fn fn_taking_struct_b(&self, struct_b: &StructB) -> bool {
        true
    }
}

async fn get_struct_a_async() -> StructA {
    StructA {}
}

async fn ice() {
    match Some(StructB {}) {
        Some(struct_b) if get_struct_a_async().await.fn_taking_struct_b(&struct_b) => {}
        _ => {}
    }
}
