use crate::spec::{LinkerFlavor, Target};

pub fn target() -> Target {
    let mut base = super::linux_musl_base::opts();
    base.cpu = "pentium4".to_string();
    base.max_atomic_width = Some(64);
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-m32".to_string());
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-Wl,-melf_i386".to_string());
    base.stack_probes = true;

    // The unwinder used by i686-unknown-linux-musl, the LLVM libunwind
    // implementation, apparently relies on frame pointers existing... somehow.
    // It's not clear to me why nor where this dependency is introduced, but the
    // test suite does not pass with frame pointers eliminated and it passes
    // with frame pointers present.
    //
    // If you think that this is no longer necessary, then please feel free to
    // ignore! If it still passes the test suite and the bots then sounds good
    // to me.
    //
    // This may or may not be related to this bug:
    // https://llvm.org/bugs/show_bug.cgi?id=30879
    base.eliminate_frame_pointer = false;

    Target {
        llvm_target: "i686-unknown-linux-musl".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 32,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            f64:32:64-f80:32-n8:16:32-S128"
            .to_string(),
        arch: "x86".to_string(),
        target_os: "linux".to_string(),
        target_env: "musl".to_string(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: base,
    }
}
