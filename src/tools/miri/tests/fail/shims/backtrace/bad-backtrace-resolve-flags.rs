#[path = "../../../utils/mod.no_std.rs"]
mod utils;

fn main() {
    unsafe {
        let mut buf = vec![std::ptr::null_mut(); utils::miri_backtrace_size(0)];

        utils::miri_get_backtrace(1, buf.as_mut_ptr());

        // miri_resolve_frame will error from an invalid backtrace before it will from invalid flags
        utils::miri_resolve_frame(buf[0], 2); //~ ERROR:  unsupported operation: unknown `miri_resolve_frame` flags 2
    }
}
