use std::borrow::Cow;

use crate::spec::{
    BinaryFormat, Cc, LinkArgs, LinkSelfContainedDefault, LinkerFlavor, Os, PanicStrategy,
    RelocModel, TargetOptions, TlsModel, add_link_args, cvs,
};

pub(crate) fn options() -> TargetOptions {
    macro_rules! args {
        ($prefix:literal) => {
            &[
                // By default LLD only gives us one page of stack (64k) which is a
                // little small. Default to a larger stack closer to other PC platforms
                // (1MB) and users can always inject their own link-args to override this.
                concat!($prefix, "-z"),
                concat!($prefix, "stack-size=1048576"),
                // By default LLD's memory layout is:
                //
                // 1. First, a blank page
                // 2. Next, all static data
                // 3. Finally, the main stack (which grows down)
                //
                // This has the unfortunate consequence that on stack overflows you
                // corrupt static data and can cause some exceedingly weird bugs. To
                // help detect this a little sooner we instead request that the stack is
                // placed before static data.
                //
                // This means that we'll generate slightly larger binaries as references
                // to static data will take more bytes in the ULEB128 encoding, but
                // stack overflow will be guaranteed to trap as it underflows instead of
                // corrupting static data.
                concat!($prefix, "--stack-first"),
                // LLD only implements C++-like demangling, which doesn't match our own
                // mangling scheme. Tell LLD to not demangle anything and leave it up to
                // us to demangle these symbols later. Currently rustc does not perform
                // further demangling, but tools like twiggy and wasm-bindgen are intended
                // to do so.
                concat!($prefix, "--no-demangle"),
            ]
        };
    }

    let mut pre_link_args = TargetOptions::link_args(LinkerFlavor::WasmLld(Cc::No), args!(""));
    add_link_args(&mut pre_link_args, LinkerFlavor::WasmLld(Cc::Yes), args!("-Wl,"));

    TargetOptions {
        is_like_wasm: true,
        binary_format: BinaryFormat::Wasm,
        families: cvs!["wasm"],

        // we allow dynamic linking, but only cdylibs. Basically we allow a
        // final library artifact that exports some symbols (a wasm module) but
        // we don't allow intermediate `dylib` crate types
        dynamic_linking: true,
        only_cdylib: true,

        // relatively self-explanatory!
        exe_suffix: ".wasm".into(),
        dll_prefix: "".into(),
        dll_suffix: ".wasm".into(),
        eh_frame_header: false,

        max_atomic_width: Some(64),

        // Unwinding doesn't work right now, so the whole target unconditionally
        // defaults to panic=abort. Note that this is guaranteed to change in
        // the future once unwinding is implemented. Don't rely on this as we're
        // basically guaranteed to change it once WebAssembly supports
        // exceptions.
        panic_strategy: PanicStrategy::Abort,

        // Wasm doesn't have atomics yet, so tell LLVM that we're in a single
        // threaded model which will legalize atomics to normal operations.
        singlethread: true,

        // we use the LLD shipped with the Rust toolchain by default
        linker: Some("rust-lld".into()),
        linker_flavor: LinkerFlavor::WasmLld(Cc::No),

        pre_link_args,

        // FIXME: Figure out cases in which WASM needs to link with a native toolchain.
        //
        // rust-lang/rust#104137: cannot blindly remove this without putting in
        // some other way to compensate for lack of `-nostartfiles` in linker
        // invocation.
        link_self_contained: LinkSelfContainedDefault::True,

        // This has no effect in LLVM 8 or prior, but in LLVM 9 and later when
        // PIC code is implemented this has quite a drastic effect if it stays
        // at the default, `pic`. In an effort to keep wasm binaries as minimal
        // as possible we're defaulting to `static` for now, but the hope is
        // that eventually we can ship a `pic`-compatible standard library which
        // works with `static` as well (or works with some method of generating
        // non-relative calls and such later on).
        relocation_model: RelocModel::Static,

        // When the atomics feature is activated then these two keys matter,
        // otherwise they're basically ignored by the standard library. In this
        // mode, however, the `#[thread_local]` attribute works (i.e.
        // `has_thread_local`) and we need to get it to work by specifying
        // `local-exec` as that's all that's implemented in LLVM today for wasm.
        has_thread_local: true,
        tls_model: TlsModel::LocalExec,

        // gdb scripts don't work on wasm blobs
        emit_debug_gdb_scripts: false,

        // There's more discussion of this at
        // https://bugs.llvm.org/show_bug.cgi?id=52442 but the general result is
        // that this isn't useful for wasm and has tricky issues with
        // representation, so this is disabled.
        generate_arange_section: false,

        // Differ from LLVM's default to use the legacy exception-handling
        // proposal instructions and use the standard exception-handling
        // instructions. Note that this is only applicable when unwinding is
        // actually turned on, which it's not by default on this target. For
        // `-Zbuild-std` builds, however, this affects when rebuilding libstd
        // with unwinding.
        llvm_args: cvs!["-wasm-use-legacy-eh=false"],

        // WASI's `sys::args::init` function ignores its arguments; instead,
        // `args::args()` makes the WASI API calls itself.
        //
        // Other Wasm targets make no use of `std::env` entirely.
        // Emscripten enables it explicitly.
        main_needs_argc_argv: false,

        ..Default::default()
    }
}

pub(crate) fn emscripten_options() -> TargetOptions {
    TargetOptions {
        os: Os::Emscripten,
        linker_flavor: LinkerFlavor::EmCc,
        // emcc emits two files - a .js file to instantiate the wasm and supply platform
        // functionality, and a .wasm file.
        exe_suffix: ".js".into(),
        linker: None,
        // Reset flags for non-Em flavors back to empty to satisfy sanity checking tests.
        pre_link_args: LinkArgs::new(),
        post_link_args: TargetOptions::link_args(LinkerFlavor::EmCc, &["-sABORTING_MALLOC=0"]),
        relocation_model: RelocModel::Pic,
        // crt_static should always be true for an executable and always false
        // for a shared library. There is no easy way to indicate this and it
        // doesn't seem to matter much so we set crt_static_allows_dylibs to
        // true and leave crt_static as true when linking dynamic libraries.
        // wasi also sets crt_static_allows_dylibs: true so this is at least
        // aligned between wasm targets.
        crt_static_respected: true,
        crt_static_default: true,
        crt_static_allows_dylibs: true,
        main_needs_argc_argv: true,
        // Use the wasm C-ABI entry name from the tool-conventions BasicCABI
        // spec rather than a raw `main`, as referenced by emscripten's crt/libc.
        // Required for entry paths like `-sPROXY_TO_PTHREAD`, whose
        // `crt1_proxy_main` links against `__main_argc_argv`.
        entry_name: "__main_argc_argv".into(),
        panic_strategy: PanicStrategy::Unwind,
        no_default_libraries: false,
        families: cvs!["unix", "wasm"],
        // Explicitly override the `base::wasm`'s `llvm_args` back to empty. The
        // base is to force using the most standard exception-handling
        // instructions, when enabled, but this target is intended to follow
        // Emscripten, which is whatever LLVM defaults to.
        llvm_args: cvs![],
        ..options()
    }
}

pub(crate) fn default_wasm64_features() -> Cow<'static, str> {
    // Any engine that implements wasm64 will surely implement the rest of
    // these features since they were all merged into the official spec by the
    // time wasm64 was designed.
    "+bulk-memory,+mutable-globals,+sign-ext,+nontrapping-fptoint".into()
}
