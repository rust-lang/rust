//! The edition of the Rust language used in a crate.
// This should live in a separate crate because we use it in both actual code and codegen.
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Edition {
    // The syntax context stuff needs the discriminants to start from 0 and be consecutive.
    Edition2015 = 0,
    Edition2018,
    Edition2021,
    Edition2024,
}

impl Edition {
    pub const DEFAULT: Edition = Edition::Edition2015;
    pub const LATEST: Edition = Edition::Edition2024;
    pub const CURRENT: Edition = Edition::Edition2024;
    /// The current latest stable edition, note this is usually not the right choice in code.
    pub const CURRENT_FIXME: Edition = Edition::Edition2024;

    pub fn from_u32(u32: u32) -> Edition {
        match u32 {
            0 => Edition::Edition2015,
            1 => Edition::Edition2018,
            2 => Edition::Edition2021,
            3 => Edition::Edition2024,
            _ => panic!("invalid edition"),
        }
    }

    pub fn at_least_2024(self) -> bool {
        self >= Edition::Edition2024
    }

    pub fn at_least_2021(self) -> bool {
        self >= Edition::Edition2021
    }

    pub fn at_least_2018(self) -> bool {
        self >= Edition::Edition2018
    }

    pub fn number(&self) -> usize {
        match self {
            Edition::Edition2015 => 2015,
            Edition::Edition2018 => 2018,
            Edition::Edition2021 => 2021,
            Edition::Edition2024 => 2024,
        }
    }

    pub fn iter() -> impl Iterator<Item = Edition> {
        [Edition::Edition2015, Edition::Edition2018, Edition::Edition2021, Edition::Edition2024]
            .iter()
            .copied()
    }
}

#[derive(Debug)]
pub struct ParseEditionError {
    invalid_input: String,
}

impl std::error::Error for ParseEditionError {}
impl fmt::Display for ParseEditionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid edition: {:?}", self.invalid_input)
    }
}

impl std::str::FromStr for Edition {
    type Err = ParseEditionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let res = match s {
            "2015" => Edition::Edition2015,
            "2018" => Edition::Edition2018,
            "2021" => Edition::Edition2021,
            "2024" => Edition::Edition2024,
            _ => return Err(ParseEditionError { invalid_input: s.to_owned() }),
        };
        Ok(res)
    }
}

impl fmt::Display for Edition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Edition::Edition2015 => "2015",
            Edition::Edition2018 => "2018",
            Edition::Edition2021 => "2021",
            Edition::Edition2024 => "2024",
        })
    }
}
