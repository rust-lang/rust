use serde::Serialize;

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum Edition {
    Edition2015,
    Edition2018,
    Edition2021,
    Edition2024,
}
