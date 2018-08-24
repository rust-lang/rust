use fmt;
use io::prelude::*;
use sys::stdio::{Stderr, stderr_prints_nothing};
use thread;

pub fn dumb_print(args: fmt::Arguments) {
    if stderr_prints_nothing() {
        return
    }
    let _ = Stderr::new().map(|mut stderr| stderr.write_fmt(args));
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
