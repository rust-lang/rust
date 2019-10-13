pub struct VTable{
    state:extern fn(),
}

impl VTable{
    pub const fn vtable()->&'static VTable{
        Self::VTABLE
    }

    const VTABLE: &'static VTable =
        &VTable{state};
}

extern fn state() {}
