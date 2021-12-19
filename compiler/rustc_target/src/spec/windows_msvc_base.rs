use crate::spec::TargetOptions;

pub fn opts() -> TargetOptions {
    let base = super::msvc_base::opts();

    TargetOptions {
        os: "windows".to_string(),
        env: "msvc".to_string(),
        vendor: "pc".to_string(),
        dynamic_linking: true,
        dll_prefix: String::new(),
        dll_suffix: ".dll".to_string(),
        exe_suffix: ".exe".to_string(),
        staticlib_prefix: String::new(),
        staticlib_suffix: ".lib".to_string(),
        families: vec!["windows".to_string()],
        crt_static_allows_dylibs: true,
        crt_static_respected: true,
        requires_uwtable: true,
        // Currently we don't pass the /NODEFAULTLIB flag to the linker on MSVC
        // as there's been trouble in the past of linking the C++ standard
        // library required by LLVM. This likely needs to happen one day, but
        // in general Windows is also a more controlled environment than
        // Unix, so it's not necessarily as critical that this be implemented.
        //
        // Note that there are also some licensing worries about statically
        // linking some libraries which require a specific agreement, so it may
        // not ever be possible for us to pass this flag.
        no_default_libraries: false,
        has_thread_local: true,

        ..base
    }
}
