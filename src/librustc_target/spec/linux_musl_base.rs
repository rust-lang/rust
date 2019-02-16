use crate::spec::{LinkerFlavor, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut base = super::linux_base::opts();

    // Make sure that the linker/gcc really don't pull in anything, including
    // default objects, libs, etc.
    base.pre_link_args_crt.insert(LinkerFlavor::Gcc, Vec::new());
    base.pre_link_args_crt.get_mut(&LinkerFlavor::Gcc).unwrap().push("-nostdlib".to_string());

    // At least when this was tested, the linker would not add the
    // `GNU_EH_FRAME` program header to executables generated, which is required
    // when unwinding to locate the unwinding information. I'm not sure why this
    // argument is *not* necessary for normal builds, but it can't hurt!
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-Wl,--eh-frame-hdr".to_string());

    // When generating a statically linked executable there's generally some
    // small setup needed which is listed in these files. These are provided by
    // a musl toolchain and are linked by default by the `musl-gcc` script. Note
    // that `gcc` also does this by default, it just uses some different files.
    //
    // Each target directory for musl has these object files included in it so
    // they'll be included from there.
    base.pre_link_objects_exe_crt.push("crt1.o".to_string());
    base.pre_link_objects_exe_crt.push("crti.o".to_string());
    base.post_link_objects_crt.push("crtn.o".to_string());

    // These targets statically link libc by default
    base.crt_static_default = true;
    // These targets allow the user to choose between static and dynamic linking.
    base.crt_static_respected = true;

    base
}
