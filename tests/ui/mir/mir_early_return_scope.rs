// run-pass
#![allow(unused_variables)]
static mut DROP: bool = false;

struct ConnWrap(Conn);
impl ::std::ops::Deref for ConnWrap {
    type Target=Conn;
    fn deref(&self) -> &Conn { &self.0 }
}

struct Conn;
impl Drop for  Conn {
    fn drop(&mut self) { unsafe { DROP = true; } }
}

fn inner() {
    let conn = &*match Some(ConnWrap(Conn)) {
        Some(val) => val,
        None => return,
    };
    return;
}

fn main() {
    inner();
    unsafe {
        assert_eq!(DROP, true);
    }
}
