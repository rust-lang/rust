//@ build-pass (FIXME(62277): could be check-pass?)
// #55266
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

struct VTable<DST> {
    _to_dst_ptr: fn(*mut ()) -> *mut DST,
}

trait HasVTableFor<DST: 'static> {
    const VTABLE: &'static VTable<DST>;
}

impl<T, DST: 'static> HasVTableFor<DST> for T {
    const VTABLE: &'static VTable<DST> = &VTable {
        _to_dst_ptr: |_: *mut ()| unsafe { std::mem::zeroed() },
    };
}

pub fn push<DST: 'static, T>() {
    <T as HasVTableFor<DST>>::VTABLE;
}

fn main() {}
