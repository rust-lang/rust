use std::str::FromStr;
use std::num::ParseIntError;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    parts: [u32; 3],
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&format!("{}.{}.{}", self.parts[0], self.parts[1], self.parts[2]))
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ParseVersionError {
    ParseIntError(ParseIntError),
    WrongNumberOfParts,
}

impl From<ParseIntError> for ParseVersionError {
    fn from(err: ParseIntError) -> Self {
        ParseVersionError::ParseIntError(err)
    }
}

impl FromStr for Version {
    type Err = ParseVersionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut iter = s.split('.').map(|part| Ok(part.parse()?));

        let mut part = || {
            iter.next()
                .unwrap_or(Err(ParseVersionError::WrongNumberOfParts))
        };

        let parts = [part()?, part()?, part()?];

        if let Some(_) = iter.next() {
            // Ensure we don't have more than 3 parts.
            return Err(ParseVersionError::WrongNumberOfParts);
        }

        Ok(Self { parts })
    }
}

#[cfg(test)]
mod test {
    use super::Version;

    #[test]
    fn test_try_from_invalid_version() {
        assert!("".parse::<Version>().is_err());
        assert!("hello".parse::<Version>().is_err());
        assert!("1.32.hi".parse::<Version>().is_err());
        assert!("1.32..1".parse::<Version>().is_err());
        assert!("1.32".parse::<Version>().is_err());
        assert!("1.32.0.1".parse::<Version>().is_err());
    }

    #[test]
    fn test_try_from_single() {
        assert_eq!("1.32.0".parse(), Ok(Version { parts: [1, 32, 0] }));
        assert_eq!("1.0.0".parse(), Ok(Version { parts: [1, 0, 0] }));
    }

    #[test]
    fn test_compare() {
        let v_1_0_0 = "1.0.0".parse::<Version>().unwrap();
        let v_1_32_0 = "1.32.0".parse::<Version>().unwrap();
        let v_1_32_1 = "1.32.1".parse::<Version>().unwrap();
        assert!(v_1_0_0 < v_1_32_1);
        assert!(v_1_0_0 < v_1_32_0);
        assert!(v_1_32_0 < v_1_32_1);
    }

    #[test]
    fn test_to_string() {
        let v_1_0_0 = "1.0.0".parse::<Version>().unwrap();
        let v_1_32_1 = "1.32.1".parse::<Version>().unwrap();

        assert_eq!(v_1_0_0.to_string(), "1.0.0");
        assert_eq!(v_1_32_1.to_string(), "1.32.1");
        assert_eq!(format!("{:<8}", v_1_32_1), "1.32.1  ");
        assert_eq!(format!("{:>8}", v_1_32_1), "  1.32.1");
    }
}
