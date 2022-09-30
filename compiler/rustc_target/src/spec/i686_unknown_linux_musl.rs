use crate::spec::{FramePointer, LinkerFlavor, StackProbeType, Target};

pub fn target() -> Target {
    let mut base = super::linux_musl_base::opts();
    base.cpu = "pentium4".into();
    base.max_atomic_width = Some(64);
    base.add_pre_link_args(LinkerFlavor::Gcc, &["-m32", "-Wl,-melf_i386"]);
    base.stack_probes = StackProbeType::X86;

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
    base.frame_pointer = FramePointer::Always;

    Target {
        llvm_target: "i686-unknown-linux-musl".into(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            f64:32:64-f80:32-n8:16:32-S128"
            .into(),
        arch: "x86".into(),
        options: base,
    }
}
