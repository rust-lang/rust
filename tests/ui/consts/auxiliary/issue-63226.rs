pub struct VTable{
    state:extern "C" fn(),
}

impl VTable{
    pub const fn vtable()->&'static VTable{
        Self::VTABLE
    }

    const VTABLE: &'static VTable =
        &VTable{state};
}

extern "C" fn state() {}
