#[derive(Clone, Copy, PartialEq)]
pub struct HasPrivateField {
    not_pub: (),
}

pub const HAS_PRIVATE_FIELD: HasPrivateField = HasPrivateField { not_pub: () };
