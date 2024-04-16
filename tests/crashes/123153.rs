//@ known-bug: #123153
pub struct wl_interface {
    pub version: str,
}

pub struct Interface {
    pub other_interfaces: &'static [&'static Interface],
    pub c_ptr: Option<&'static wl_interface>,
}

pub static mut wl_callback_interface: wl_interface = wl_interface { version: 0 };

pub static WL_CALLBACK_INTERFACE: Interface =
    Interface { other_interfaces: &[], c_ptr: Some(unsafe { &wl_callback_interface }) };


fn main() {}
