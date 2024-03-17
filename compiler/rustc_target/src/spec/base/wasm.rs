use crate::spec::{
    add_link_args, cvs, Cc, LinkSelfContainedDefault, LinkerFlavor, MaybeLazy, PanicStrategy,
    RelocModel, TargetOptions, TlsModel,
};

pub fn options() -> TargetOptions {
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
                // FIXME we probably shouldn't pass this but instead pass an explicit list
                // of symbols we'll allow to be undefined. We don't currently have a
                // mechanism of knowing, however, which symbols are intended to be imported
                // from the environment and which are intended to be imported from other
                // objects linked elsewhere. This is a coarse approximation but is sure to
                // hide some bugs and frustrate someone at some point, so we should ideally
                // work towards a world where we can explicitly list symbols that are
                // supposed to be imported and have all other symbols generate errors if
                // they remain undefined.
                concat!($prefix, "--allow-undefined"),
                // LLD only implements C++-like demangling, which doesn't match our own
                // mangling scheme. Tell LLD to not demangle anything and leave it up to
                // us to demangle these symbols later. Currently rustc does not perform
                // further demangling, but tools like twiggy and wasm-bindgen are intended
                // to do so.
                concat!($prefix, "--no-demangle"),
            ]
        };
    }

    let pre_link_args = MaybeLazy::lazy(|| {
        let mut pre_link_args = TargetOptions::link_args(LinkerFlavor::WasmLld(Cc::No), args!(""));
        add_link_args(&mut pre_link_args, LinkerFlavor::WasmLld(Cc::Yes), args!("-Wl,"));
        pre_link_args
    });

    TargetOptions {
        is_like_wasm: true,
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

        // Symbol visibility takes care of this for the WebAssembly.
        // Additionally the only known linker, LLD, doesn't support the script
        // arguments just yet
        limit_rdylib_exports: false,

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

        ..Default::default()
    }
}
