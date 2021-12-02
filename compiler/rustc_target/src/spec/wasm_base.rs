use super::crt_objects::CrtObjectsFallback;
use super::{LinkerFlavor, LldFlavor, PanicStrategy, RelocModel, TargetOptions, TlsModel};
use std::collections::BTreeMap;

pub fn options() -> TargetOptions {
    let mut lld_args = Vec::new();
    let mut clang_args = Vec::new();
    let mut arg = |arg: &str| {
        lld_args.push(arg.to_string());
        clang_args.push(format!("-Wl,{}", arg));
    };

    // By default LLD only gives us one page of stack (64k) which is a
    // little small. Default to a larger stack closer to other PC platforms
    // (1MB) and users can always inject their own link-args to override this.
    arg("-z");
    arg("stack-size=1048576");

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
    arg("--stack-first");

    // FIXME we probably shouldn't pass this but instead pass an explicit list
    // of symbols we'll allow to be undefined. We don't currently have a
    // mechanism of knowing, however, which symbols are intended to be imported
    // from the environment and which are intended to be imported from other
    // objects linked elsewhere. This is a coarse approximation but is sure to
    // hide some bugs and frustrate someone at some point, so we should ideally
    // work towards a world where we can explicitly list symbols that are
    // supposed to be imported and have all other symbols generate errors if
    // they remain undefined.
    arg("--allow-undefined");

    // Rust code should never have warnings, and warnings are often
    // indicative of bugs, let's prevent them.
    arg("--fatal-warnings");

    // LLD only implements C++-like demangling, which doesn't match our own
    // mangling scheme. Tell LLD to not demangle anything and leave it up to
    // us to demangle these symbols later. Currently rustc does not perform
    // further demangling, but tools like twiggy and wasm-bindgen are intended
    // to do so.
    arg("--no-demangle");

    let mut pre_link_args = BTreeMap::new();
    pre_link_args.insert(LinkerFlavor::Lld(LldFlavor::Wasm), lld_args);
    pre_link_args.insert(LinkerFlavor::Gcc, clang_args);

    TargetOptions {
        is_like_wasm: true,
        families: vec!["wasm".to_string()],

        // we allow dynamic linking, but only cdylibs. Basically we allow a
        // final library artifact that exports some symbols (a wasm module) but
        // we don't allow intermediate `dylib` crate types
        dynamic_linking: true,
        only_cdylib: true,

        // This means we'll just embed a `#[start]` function in the wasm module
        executables: true,

        // relatively self-explanatory!
        exe_suffix: ".wasm".to_string(),
        dll_prefix: String::new(),
        dll_suffix: ".wasm".to_string(),
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

        // no dynamic linking, no need for default visibility!
        default_hidden_visibility: true,

        // Symbol visibility takes care of this for the WebAssembly.
        // Additionally the only known linker, LLD, doesn't support the script
        // arguments just yet
        limit_rdylib_exports: false,

        // we use the LLD shipped with the Rust toolchain by default
        linker: Some("rust-lld".to_owned()),
        lld_flavor: LldFlavor::Wasm,
        linker_is_gnu: false,

        pre_link_args,

        crt_objects_fallback: Some(CrtObjectsFallback::Wasm),

        // This has no effect in LLVM 8 or prior, but in LLVM 9 and later when
        // PIC code is implemented this has quite a drastric effect if it stays
        // at the default, `pic`. In an effort to keep wasm binaries as minimal
        // as possible we're defaulting to `static` for now, but the hope is
        // that eventually we can ship a `pic`-compatible standard library which
        // works with `static` as well (or works with some method of generating
        // non-relative calls and such later on).
        relocation_model: RelocModel::Static,

        // When the atomics feature is activated then these two keys matter,
        // otherwise they're basically ignored by the standard library. In this
        // mode, however, the `#[thread_local]` attribute works (i.e.
        // `has_elf_tls`) and we need to get it to work by specifying
        // `local-exec` as that's all that's implemented in LLVM today for wasm.
        has_elf_tls: true,
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
