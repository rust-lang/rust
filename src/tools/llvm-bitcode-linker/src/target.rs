#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, clap::ValueEnum)]
pub enum Target {
    Nvptx64NvidiaCuda,
}

#[derive(Debug, Clone, Copy, thiserror::Error)]
/// The target is not supported by this linker
#[error("unsupported target")]
pub struct UnsupportedTarget;

impl std::str::FromStr for Target {
    type Err = UnsupportedTarget;

    fn from_str(s: &str) -> Result<Target, UnsupportedTarget> {
        match s {
            "nvptx64-nvidia-cuda" => Ok(Target::Nvptx64NvidiaCuda),
            _ => Err(UnsupportedTarget),
        }
    }
}
