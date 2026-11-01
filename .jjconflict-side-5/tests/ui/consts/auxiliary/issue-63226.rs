pub struct VTable{
    state:extern "C" fn(),
}

impl VTable {
    pub const fn vtable()->&'static VTable{
        Self::VTABLE
    }

    const VTABLE: &'static VTable =
        &VTable{state};

    pub const VTABLE2: &'static VTable =
        &VTable{state: state2};
}

pub const VTABLE3: &'static VTable =
    &VTable{state: state3};

// Only referenced via a `pub const fn`, and yet reachable.
extern "C" fn state() {}
// Only referenced via a associated `pub const`, and yet reachable.
extern "C" fn state2() {}
// Only referenced via a free `pub const`, and yet reachable.
extern "C" fn state3() {}
