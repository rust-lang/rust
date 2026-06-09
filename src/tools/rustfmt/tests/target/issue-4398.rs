impl Struct {
    /// Documentation for `foo`
    #[rustfmt::skip] // comment on why use a skip here
    pub fn foo(&self) {}
}

impl Struct {
    /// Documentation for `foo`
    #[rustfmt::skip] // comment on why use a skip here
    pub fn foo(&self) {}
}

/// Documentation for `Struct`
#[rustfmt::skip] // comment
impl Struct {
    /// Documentation for `foo`
       #[rustfmt::skip] // comment on why use a skip here
    pub fn foo(&self) {}
}
