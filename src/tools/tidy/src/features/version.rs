use std::str::FromStr;
use std::num::ParseIntError;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    parts: Vec<u32>,
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let x = self.parts.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(".");
        f.pad(&x)
    }
}

impl FromStr for Version {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts = s.split('.').map(|part| part.parse()).collect::<Result<_, _>>()?;
        Ok(Version { parts })
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
    }

    #[test]
    fn test_try_from_single() {
        assert_eq!("1.32.0".parse(), Ok(Version { parts: vec![1, 32, 0] }));
        assert_eq!("1.0.0".parse(), Ok(Version { parts: vec![1, 0, 0] }));
    }

    #[test]
    fn test_compare() {
        let v_1_0_0 = "1.0.0".parse::<Version>().unwrap();
        let v_1_32 = "1.32".parse::<Version>().unwrap();
        let v_1_32_1 = "1.32.1".parse::<Version>().unwrap();
        assert!(v_1_0_0 < v_1_32_1);
        assert!(v_1_0_0 < v_1_32);
        assert!(v_1_32 < v_1_32_1);
    }

    #[test]
    fn test_to_string() {
        let v_1_0_0 = "1.0.0".parse::<Version>().unwrap();
        let v_1_32 = "1.32".parse::<Version>().unwrap();
        let v_1_32_1 = "1.32.1".parse::<Version>().unwrap();

        assert_eq!(v_1_0_0.to_string(), "1.0.0");
        assert_eq!(v_1_32.to_string(), "1.32");
        assert_eq!(v_1_32_1.to_string(), "1.32.1");
        assert_eq!(format!("{:<8}", v_1_32_1), "1.32.1  ");
        assert_eq!(format!("{:>8}", v_1_32_1), "  1.32.1");
    }
}
