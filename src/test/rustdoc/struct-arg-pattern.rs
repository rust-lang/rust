#![crate_name = "foo"]

struct BodyId {
    hir_id: usize,
}

// @has 'foo/fn.body_owner.html' '//*[@class="rust fn"]' 'pub fn body_owner(_: BodyId)'
pub fn body_owner(BodyId { hir_id }: BodyId) {
    // ...
}
