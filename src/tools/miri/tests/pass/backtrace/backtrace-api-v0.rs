//@normalize-stderr-test: "::<.*>" -> ""

#[inline(never)]
fn func_a() -> Box<[*mut ()]> {
    func_b::<u8>()
}
#[inline(never)]
fn func_b<T>() -> Box<[*mut ()]> {
    func_c()
}

macro_rules! invoke_func_d {
    () => {
        func_d()
    };
}

#[inline(never)]
fn func_c() -> Box<[*mut ()]> {
    invoke_func_d!()
}
#[inline(never)]
fn func_d() -> Box<[*mut ()]> {
    unsafe { miri_get_backtrace(0) }
}

fn main() {
    let mut seen_main = false;
    let frames = func_a();
    for frame in frames.into_iter() {
        let miri_frame = unsafe { miri_resolve_frame(*frame, 0) };
        let name = String::from_utf8(miri_frame.name.into()).unwrap();
        let filename = String::from_utf8(miri_frame.filename.into()).unwrap();

        if name == "func_a" {
            assert_eq!(func_a as *mut (), miri_frame.fn_ptr);
        }

        // Print every frame to stderr.
        let out = format!("{}:{}:{} ({})", filename, miri_frame.lineno, miri_frame.colno, name);
        eprintln!("{}", out);
        // Print the 'main' frame (and everything before it) to stdout, skipping
        // the printing of internal (and possibly fragile) libstd frames.
        if !seen_main {
            println!("{}", out);
            seen_main = name == "main";
        }
    }
}

// This goes at the bottom of the file so that we can change it
// without disturbing line numbers of the functions in the backtrace.

extern "Rust" {
    fn miri_get_backtrace(flags: u64) -> Box<[*mut ()]>;
    fn miri_resolve_frame(ptr: *mut (), flags: u64) -> MiriFrame;
}

#[derive(Debug)]
#[repr(C)]
struct MiriFrame {
    name: Box<[u8]>,
    filename: Box<[u8]>,
    lineno: u32,
    colno: u32,
    fn_ptr: *mut (),
}
