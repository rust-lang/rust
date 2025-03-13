use crate::spec::TargetOptions;

pub(crate) fn opts() -> TargetOptions{
    TargetOptions{
        os: "hobkey".into(),
        plt_by_default: false,
        max_atomic_width: Some(64),
        has_thread_local: false,
        ..Default::default()
    }
}