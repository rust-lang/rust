pub struct HasPrivateFields {
    _private: (),
}

impl HasPrivateFields {
    pub const ASSOC: Self = Self { _private: () };
}
