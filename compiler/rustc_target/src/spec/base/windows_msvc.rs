use crate::spec::{TargetOptions, base, cvs};

pub(crate) fn opts() -> TargetOptions {
    let base = base::msvc::opts();

    TargetOptions {
        os: "windows".into(),
        env: "msvc".into(),
        vendor: "pc".into(),
        dynamic_linking: true,
        dll_prefix: "".into(),
        dll_suffix: ".dll".into(),
        exe_suffix: ".exe".into(),
        staticlib_prefix: "".into(),
        staticlib_suffix: ".lib".into(),
        families: cvs!["windows"],
        crt_static_allows_dylibs: true,
        crt_static_respected: true,
        requires_uwtable: true,
        // We don't pass the /NODEFAULTLIB flag to the linker on MSVC
        // as that prevents linker directives embedded in object files from
        // including other necessary libraries.
        //
        // For example, msvcrt.lib embeds a linker directive like:
        //    /DEFAULTLIB:vcruntime.lib /DEFAULTLIB:ucrt.lib
        // So that vcruntime.lib and ucrt.lib are included when the entry point
        // in msvcrt.lib is used. Using /NODEFAULTLIB would mean having to
        // manually add those two libraries and potentially further dependencies
        // they bring in.
        //
        // See also https://learn.microsoft.com/en-us/cpp/preprocessor/comment-c-cpp?view=msvc-170#lib
        // for documentation on including library dependencies in C/C++ code.
        no_default_libraries: false,
        has_thread_local: true,

        ..base
    }
}
