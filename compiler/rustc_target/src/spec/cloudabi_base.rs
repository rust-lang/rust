use crate::spec::{LinkArgs, LinkerFlavor, RelroLevel, TargetOptions, TlsModel};

pub fn opts() -> TargetOptions {
    let mut args = LinkArgs::new();
    args.insert(
        LinkerFlavor::Gcc,
        vec![
            "-Wl,-Bstatic".to_string(),
            "-Wl,--no-dynamic-linker".to_string(),
            "-Wl,--gc-sections".to_string(),
        ],
    );

    TargetOptions {
        target_os: "cloudabi".to_string(),
        executables: true,
        target_family: None,
        linker_is_gnu: true,
        pre_link_args: args,
        position_independent_executables: true,
        // As CloudABI only supports static linkage, there is no need
        // for dynamic TLS. The C library therefore does not provide
        // __tls_get_addr(), which is normally used to perform dynamic
        // TLS lookups by programs that make use of dlopen(). Only the
        // "local-exec" and "initial-exec" TLS models can be used.
        //
        // "local-exec" is more efficient than "initial-exec", as the
        // latter has one more level of indirection: it accesses the GOT
        // (Global Offset Table) to obtain the effective address of a
        // thread-local variable. Using a GOT is useful only when doing
        // dynamic linking.
        tls_model: TlsModel::LocalExec,
        relro_level: RelroLevel::Full,
        ..Default::default()
    }
}
