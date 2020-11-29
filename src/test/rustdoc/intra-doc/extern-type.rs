#![feature(extern_types)]

extern {
    pub type ExternType;
}

impl ExternType {
    pub fn f(&self) {

    }
}

// @has 'extern_type/foreigntype.ExternType.html'
// @has 'extern_type/fn.links_to_extern_type.html' \
// 'href="../extern_type/foreigntype.ExternType.html#method.f"'
/// See also [ExternType::f]
pub fn links_to_extern_type() {}
