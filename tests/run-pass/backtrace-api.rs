// normalize-stderr-test ".*rustlib" -> "RUSTLIB"
// normalize-stderr-test "RUSTLIB/(.*):\d+:\d+ "-> "RUSTLIB/$1:LL:COL "

extern "Rust" {
    fn miri_get_backtrace() -> Box<[*mut ()]>;
    fn miri_resolve_frame(version: u8, ptr: *mut ()) -> MiriFrame;
}

#[derive(Debug)]
struct MiriFrame {
    name: Box<[u8]>,
    filename: Box<[u8]>,
    lineno: u32,
    colno: u32
}

fn main() {
    let frames = unsafe { miri_get_backtrace() };
    for frame in frames.into_iter() {
        let miri_frame = unsafe { miri_resolve_frame(0, *frame) };
        let name = String::from_utf8(miri_frame.name.into()).unwrap();
        let filename = String::from_utf8(miri_frame.filename.into()).unwrap();
        eprintln!("{}:{}:{} ({})", filename, miri_frame.lineno, miri_frame.colno, name);
    }
}
