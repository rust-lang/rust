#![crate_name = "closure_dep"]
#![crate_type = "rlib"]

#[cfg(rpass1)]
struct PrivateData {
    value: u32,
}

#[cfg(any(rpass2, rpass3))]
struct PrivateData {
    value: u32,
    _extra: u32,
}

#[cfg(rpass1)]
fn make_private() -> PrivateData {
    PrivateData { value: 42 }
}

#[cfg(any(rpass2, rpass3))]
fn make_private() -> PrivateData {
    PrivateData { value: 42, _extra: 0 }
}

pub fn make_closure() -> impl Fn() -> u32 {
    let data = make_private();
    move || data.value
}

pub fn call_with_closure<F: Fn(u32) -> u32>(f: F, x: u32) -> u32 {
    let private = make_private();
    f(x) + private.value
}
