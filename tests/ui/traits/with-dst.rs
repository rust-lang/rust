// build-pass (FIXME(62277): could be check-pass?)
// #55266

struct VTable<DST: ?Sized> {
    _to_dst_ptr: fn(*mut ()) -> *mut DST,
}

trait HasVTableFor<DST: ?Sized + 'static> {
    const VTABLE: &'static VTable<DST>;
}

impl<T, DST: ?Sized + 'static> HasVTableFor<DST> for T {
    const VTABLE: &'static VTable<DST> = &VTable {
        _to_dst_ptr: |_: *mut ()| unsafe { std::mem::zeroed() },
    };
}

pub fn push<DST: ?Sized + 'static, T>() {
    <T as HasVTableFor<DST>>::VTABLE;
}

fn main() {}
