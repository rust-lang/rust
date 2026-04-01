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
    unsafe {
        let count = miri_backtrace_size(0);
        let mut buf = vec![std::ptr::null_mut(); count];
        miri_get_backtrace(1, buf.as_mut_ptr());
        buf.into()
    }
}

fn main() {
    let mut seen_main = false;
    let frames = func_a();
    for frame in frames.iter() {
        let miri_frame = unsafe { miri_resolve_frame(*frame, 1) };

        let mut name = vec![0; miri_frame.name_len];
        let mut filename = vec![0; miri_frame.filename_len];

        unsafe {
            miri_resolve_frame_names(*frame, 0, name.as_mut_ptr(), filename.as_mut_ptr());
        }

        let name = String::from_utf8(name).unwrap();
        let filename = String::from_utf8(filename).unwrap();

        if name == "func_a" {
            assert_eq!(func_a as *mut (), miri_frame.fn_ptr);
        }

        // Print every frame to stderr.
        let out = format!("{}:{}:{} ({})", filename, miri_frame.lineno, miri_frame.colno, name);
        eprintln!("{}", out);
        // Print the 'main' frame (and everything before it) to stdout, skipping
        // the printing of internal (and possibly fragile) libstd frames.
        // Stdout is less normalized so we see more, but it also means we can print less
        // as platform differences would lead to test suite failures.
        if !seen_main {
            println!("{}", out);
            seen_main = name == "main";
        }
    }
}

// This goes at the bottom of the file so that we can change it
// without disturbing line numbers of the functions in the backtrace.

extern "Rust" {
    fn miri_backtrace_size(flags: u64) -> usize;
    fn miri_get_backtrace(flags: u64, buf: *mut *mut ());
    fn miri_resolve_frame(ptr: *mut (), flags: u64) -> MiriFrame;
    fn miri_resolve_frame_names(ptr: *mut (), flags: u64, name_buf: *mut u8, filename_buf: *mut u8);
}

#[derive(Debug)]
#[repr(C)]
struct MiriFrame {
    name_len: usize,
    filename_len: usize,
    lineno: u32,
    colno: u32,
    fn_ptr: *mut (),
}
