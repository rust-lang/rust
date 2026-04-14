#![no_std]
#![no_std]
extern crate alloc;

#[cfg(all(target_os = "thingos", panic = "unwind"))]
compile_error!("Thing-OS does not support panic=unwind. Use panic=abort (target contract).");

pub use abi;
#[cfg(feature = "rt")]
pub use stem_macros::main;

pub mod arch;
pub mod bitset;
pub mod block;
pub mod console;
pub mod device;
pub mod errors;
pub mod fs;

#[cfg(feature = "global-alloc")]
pub mod heap;
pub mod i18n;
#[cfg(feature = "rt")]
pub mod memory;

/// Platform Abstraction Layer - explicit platform contract
pub mod pal;
#[cfg(not(feature = "std"))]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    crate::error!("PANIC: {}", info);
    crate::syscall::exit(101);
}
pub mod pci;
pub mod perf;
pub mod rt;
pub mod simd;
pub mod stack;
pub mod syscall;
pub mod task;
pub mod thread;
pub mod time;
pub mod tls;
pub mod utils;
pub mod vm;
pub mod wait_set;

// Re-export time types for convenience
pub use time::{Duration, Instant};

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => ($crate::console::print(format_args!($($arg)*)));
}

#[macro_export]
macro_rules! println {
    () => ($crate::print!("\n"));
    ($($arg:tt)*) => ($crate::print!("{}\n", format_args!($($arg)*)));
}

pub fn log(s: &str) {
    pal::log::write(pal::log::Level::Info, format_args!("{}", s));
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {{
        $crate::console::log_with_provenance(
            $crate::abi::logging::Level::Error as usize,
            module_path!(),
            format_args!($($arg)*),
        );
    }};
}

#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {{
        $crate::console::log_with_provenance(
            $crate::abi::logging::Level::Warn as usize,
            module_path!(),
            format_args!($($arg)*),
        );
    }};
}

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {{
        $crate::console::log_with_provenance(
            $crate::abi::logging::Level::Info as usize,
            module_path!(),
            format_args!($($arg)*),
        );
    }};
}

#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {{
        $crate::console::log_with_provenance(
            $crate::abi::logging::Level::Debug as usize,
            module_path!(),
            format_args!($($arg)*),
        );
    }};
}

#[macro_export]
macro_rules! trace {
    ($($arg:tt)*) => {{
        $crate::console::log_with_provenance(
            $crate::abi::logging::Level::Trace as usize,
            module_path!(),
            format_args!($($arg)*),
        );
    }};
}

pub fn yield_now() {
    thread::yield_now();
}

pub fn sleep(duration: impl Into<Duration>) {
    time::sleep(duration.into());
}

pub fn sleep_ms(ms: u64) {
    time::sleep_ms(ms);
}

pub fn monotonic_ns() -> u64 {
    time::monotonic_ns()
}

/// Returns the current monotonic instant.
#[inline]
pub fn now() -> Instant {
    time::now()
}

pub use syscall::{reboot, shutdown};

pub mod thing;

#[cfg(feature = "global-alloc")]
pub mod allocator;
