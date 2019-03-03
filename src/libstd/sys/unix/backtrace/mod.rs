/// Backtrace support built on libgcc with some extra OS-specific support
///
/// Some methods of getting a backtrace:
///
/// * The backtrace() functions on unix. It turns out this doesn't work very
///   well for green threads on macOS, and the address to symbol portion of it
///   suffers problems that are described below.
///
/// * Using libunwind. This is more difficult than it sounds because libunwind
///   isn't installed everywhere by default. It's also a bit of a hefty library,
///   so possibly not the best option. When testing, libunwind was excellent at
///   getting both accurate backtraces and accurate symbols across platforms.
///   This route was not chosen in favor of the next option, however.
///
/// * We're already using libgcc_s for exceptions in rust (triggering thread
///   unwinding and running destructors on the stack), and it turns out that it
///   conveniently comes with a function that also gives us a backtrace. All of
///   these functions look like _Unwind_*, but it's not quite the full
///   repertoire of the libunwind API. Due to it already being in use, this was
///   the chosen route of getting a backtrace.
///
/// After choosing libgcc_s for backtraces, the sad part is that it will only
/// give us a stack trace of instruction pointers. Thankfully these instruction
/// pointers are accurate (they work for green and native threads), but it's
/// then up to us again to figure out how to translate these addresses to
/// symbols. As with before, we have a few options. Before, that, a little bit
/// of an interlude about symbols. This is my very limited knowledge about
/// symbol tables, and this information is likely slightly wrong, but the
/// general idea should be correct.
///
/// When talking about symbols, it's helpful to know a few things about where
/// symbols are located. Some symbols are located in the dynamic symbol table
/// of the executable which in theory means that they're available for dynamic
/// linking and lookup. Other symbols end up only in the local symbol table of
/// the file. This loosely corresponds to pub and priv functions in Rust.
///
/// Armed with this knowledge, we know that our solution for address to symbol
/// translation will need to consult both the local and dynamic symbol tables.
/// With that in mind, here's our options of translating an address to
/// a symbol.
///
/// * Use dladdr(). The original backtrace()-based idea actually uses dladdr()
///   behind the scenes to translate, and this is why backtrace() was not used.
///   Conveniently, this method works fantastically on macOS. It appears dladdr()
///   uses magic to consult the local symbol table, or we're putting everything
///   in the dynamic symbol table anyway. Regardless, for macOS, this is the
///   method used for translation. It's provided by the system and easy to do.o
///
///   Sadly, all other systems have a dladdr() implementation that does not
///   consult the local symbol table. This means that most functions are blank
///   because they don't have symbols. This means that we need another solution.
///
/// * Use unw_get_proc_name(). This is part of the libunwind api (not the
///   libgcc_s version of the libunwind api), but involves taking a dependency
///   to libunwind. We may pursue this route in the future if we bundle
///   libunwind, but libunwind was unwieldy enough that it was not chosen at
///   this time to provide this functionality.
///
/// * Shell out to a utility like `readelf`. Crazy though it may sound, it's a
///   semi-reasonable solution. The stdlib already knows how to spawn processes,
///   so in theory it could invoke readelf, parse the output, and consult the
///   local/dynamic symbol tables from there. This ended up not getting chosen
///   due to the craziness of the idea plus the advent of the next option.
///
/// * Use `libbacktrace`. It turns out that this is a small library bundled in
///   the gcc repository which provides backtrace and symbol translation
///   functionality. All we really need from it is the backtrace functionality,
///   and we only really need this on everything that's not macOS, so this is the
///   chosen route for now.
///
/// In summary, the current situation uses libgcc_s to get a trace of stack
/// pointers, and we use dladdr() or libbacktrace to translate these addresses
/// to symbols. This is a bit of a hokey implementation as-is, but it works for
/// all unix platforms we support right now, so it at least gets the job done.

pub use self::tracing::unwind_backtrace;
pub use self::printing::{foreach_symbol_fileline, resolve_symname};

// tracing impls:
mod tracing;
// symbol resolvers:
mod printing;

#[cfg(not(target_os = "emscripten"))]
pub mod gnu {
    use crate::io;
    use crate::fs;

    use libc::c_char;

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub fn get_executable_filename() -> io::Result<(Vec<c_char>, fs::File)> {
        Err(io::Error::new(io::ErrorKind::Other, "Not implemented"))
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn get_executable_filename() -> io::Result<(Vec<c_char>, fs::File)> {
        use crate::env;
        use crate::os::unix::ffi::OsStrExt;

        let filename = env::current_exe()?;
        let file = fs::File::open(&filename)?;
        let mut filename_cstr: Vec<_> = filename.as_os_str().as_bytes().iter()
            .map(|&x| x as c_char).collect();
        filename_cstr.push(0); // Null terminate
        Ok((filename_cstr, file))
    }
}

pub struct BacktraceContext;
