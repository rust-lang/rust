Language #
Replace const eval limit by a lint and add an exponential backoff warning
expand: Change how #![cfg(FALSE)] behaves on crate root
Stabilize inline asm for LoongArch64
Uplift clippy::undropped_manually_drops lint
Uplift clippy::invalid_utf8_in_unchecked lint as invalid_from_utf8_unchecked and invalid_from_utf8
Uplift clippy::cast_ref_to_mut lint as invalid_reference_casting
Uplift clippy::cmp_nan lint as invalid_nan_comparisons
resolve: Remove artificial import ambiguity errors
Don’t require associated types with Self: Sized bounds in dyn Trait objects
Compiler #
Remember names of cfg-ed out items to mention them in diagnostics
Support for native WASM exceptions
Add support for NetBSD/aarch64-be (big-endian arm64).
Write to stdout if - is given as output file
Force all native libraries to be statically linked when linking a static binary
Add Tier 3 support for loongarch64-unknown-none*
Prevent .eh_frame from being emitted for -C panic=abort
Support 128-bit enum variant in debuginfo codegen
compiler: update solaris/illumos to enable tsan support.
Refer to Rust’s [platform support page][platform-support-doc] for more information on Rust’s tiered platform support.

Libraries #
Document memory orderings of thread::{park, unpark}
io: soften ‘at most one write attempt’ requirement in io::Write::write
Specify behavior of HashSet::insert
Relax implicit T: Sized bounds on BufReader<T>, BufWriter<T> and LineWriter<T>
Update runtime guarantee for select_nth_unstable
Return Ok on kill if process has already exited
Implement PartialOrd for Vecs over different allocators
Use 128 bits for TypeId hash
Don’t drain-on-drop in DrainFilter impls of various collections.
Make {Arc,Rc,Weak}::ptr_eq ignore pointer metadata
Rustdoc #
Allow whitespace as path separator like double colon
Add search result item types after their name
Search for slices and arrays by type with []
Clean up type unification and “unboxing”
Stabilized APIs #
impl<T: Send> Sync for mpsc::Sender<T>
impl TryFrom<&OsStr> for &str
String::leak
These APIs are now stable in const contexts:

CStr::from_bytes_with_nul
CStr::to_bytes
CStr::to_bytes_with_nul
CStr::to_str
Cargo #
Enable -Zdoctest-in-workspace by default. When running each documentation test, the working directory is set to the root directory of the package the test belongs to. docs #12221 #12288
Add support of the “default” keyword to reset previously set build.jobs parallelism back to the default. #12222
Compatibility Notes #
Alter Display for Ipv6Addr for IPv4-compatible addresses
Cargo changed feature name validation check to a hard error. The warning was added in Rust 1.49. These extended characters aren’t allowed on crates.io, so this should only impact users of other registries, or people who don’t publish to a registry. #12291
