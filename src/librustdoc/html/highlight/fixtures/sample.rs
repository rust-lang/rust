#![crate_type = "lib"]

#[cfg(target_os = "linux")]
fn main() {
    let foo = true && false || true;
    let _: *const () = 0;
    let _ = &foo;
    let _ = &&foo;
    let _ = *foo;
    mac!(foo, &mut bar);
    assert!(self.length < N && index <= self.length);
}

macro_rules! bar {
    ($foo:tt) => {};
}
