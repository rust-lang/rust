use crate::spec::{PanicStrategy, RelroLevel, TargetOptions};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: "helenos".into(),

        dynamic_linking: true,
        // FIXME: this actually is supported by HelenOS, but then we run into issues
        // with linking libstartfiles.a (parts of which obviously can't be at randomized
        // positions). The crt_* flags also have some effect on this.
        // position_independent_executables: true,

        relro_level: RelroLevel::Full,
        panic_strategy: PanicStrategy::Abort,

        ..Default::default()
    }
}
