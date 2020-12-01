#![feature(extern_types)]

extern {
    pub type ExternType;
}

impl ExternType {
    pub fn f(&self) {

    }
}

// @has 'intra_link_extern_type/foreigntype.ExternType.html'
// @has 'intra_link_extern_type/fn.links_to_extern_type.html' \
// 'href="../intra_link_extern_type/foreigntype.ExternType.html#method.f"'
/// See also [ExternType::f]
pub fn links_to_extern_type() {
}
