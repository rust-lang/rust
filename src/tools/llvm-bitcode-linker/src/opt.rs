use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, Default, Hash, Eq, PartialEq, clap::ValueEnum)]
pub enum Optimization {
    #[default]
    #[value(name = "0")]
    O0,
    #[value(name = "1")]
    O1,
    #[value(name = "2")]
    O2,
    #[value(name = "3")]
    O3,
    #[value(name = "s")]
    Os,
    #[value(name = "z")]
    Oz,
}

#[derive(Debug, Clone, Copy, thiserror::Error)]
/// An invalid optimization level
#[error("invalid optimization level")]
pub struct InvalidOptimizationLevel;

impl std::str::FromStr for Optimization {
    type Err = InvalidOptimizationLevel;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" | "O0" => Ok(Optimization::O0),
            "1" | "O1" => Ok(Optimization::O1),
            "2" | "O2" => Ok(Optimization::O2),
            "3" | "O3" => Ok(Optimization::O3),
            "s" | "Os" => Ok(Optimization::Os),
            "z" | "Oz" => Ok(Optimization::Oz),
            _ => Err(InvalidOptimizationLevel),
        }
    }
}

impl Display for Optimization {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match *self {
            Optimization::O0 => write!(f, "O0"),
            Optimization::O1 => write!(f, "O1"),
            Optimization::O2 => write!(f, "O2"),
            Optimization::O3 => write!(f, "O3"),
            Optimization::Os => write!(f, "Os"),
            Optimization::Oz => write!(f, "Oz"),
        }
    }
}
