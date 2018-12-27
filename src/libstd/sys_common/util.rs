use fmt;
use io::prelude::*;
use sys::stdio::panic_output;
use thread;

pub fn dumb_print(args: fmt::Arguments) {
    if let Some(mut out) = panic_output() {
        let _ = out.write_fmt(args);
    }
}

// Other platforms should use the appropriate platform-specific mechanism for
// aborting the process.  If no platform-specific mechanism is available,
// ::intrinsics::abort() may be used instead.  The above implementations cover
// all targets currently supported by libstd.

pub fn abort(args: fmt::Arguments) -> ! {
    dumb_print(format_args!("fatal runtime error: {}\n", args));
    unsafe { ::sys::abort_internal(); }
}

#[allow(dead_code)] // stack overflow detection not enabled on all platforms
pub unsafe fn report_overflow() {
    dumb_print(format_args!("\nthread '{}' has overflowed its stack\n",
                            thread::current().name().unwrap_or("<unknown>")));
}
