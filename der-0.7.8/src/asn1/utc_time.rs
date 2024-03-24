//! ASN.1 `UTCTime` support.

use crate::{
    datetime::{self, DateTime},
    ord::OrdIsValueOrd,
    DecodeValue, EncodeValue, Error, ErrorKind, FixedTag, Header, Length, Reader, Result, Tag,
    Writer,
};
use core::time::Duration;

#[cfg(feature = "std")]
use std::time::SystemTime;

/// ASN.1 `UTCTime` type.
///
/// This type implements the validity requirements specified in
/// [RFC 5280 Section 4.1.2.5.1][1], namely:
///
/// > For the purposes of this profile, UTCTime values MUST be expressed in
/// > Greenwich Mean Time (Zulu) and MUST include seconds (i.e., times are
/// > `YYMMDDHHMMSSZ`), even where the number of seconds is zero.  Conforming
/// > systems MUST interpret the year field (`YY`) as follows:
/// >
/// > - Where `YY` is greater than or equal to 50, the year SHALL be
/// >   interpreted as `19YY`; and
/// > - Where `YY` is less than 50, the year SHALL be interpreted as `20YY`.
///
/// Note: Due to common operations working on `UNIX_EPOCH` [`UtcTime`]s are
/// only supported for the years 1970-2049.
///
/// [1]: https://tools.ietf.org/html/rfc5280#section-4.1.2.5.1
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct UtcTime(DateTime);

impl UtcTime {
    /// Length of an RFC 5280-flavored ASN.1 DER-encoded [`UtcTime`].
    pub const LENGTH: usize = 13;

    /// Maximum year that can be represented as a `UTCTime`.
    pub const MAX_YEAR: u16 = 2049;

    /// Create a [`UtcTime`] from a [`DateTime`].
    pub fn from_date_time(datetime: DateTime) -> Result<Self> {
        if datetime.year() <= UtcTime::MAX_YEAR {
            Ok(Self(datetime))
        } else {
            Err(Self::TAG.value_error())
        }
    }

    /// Convert this [`UtcTime`] into a [`DateTime`].
    pub fn to_date_time(&self) -> DateTime {
        self.0
    }

    /// Create a new [`UtcTime`] given a [`Duration`] since `UNIX_EPOCH`
    /// (a.k.a. "Unix time")
    pub fn from_unix_duration(unix_duration: Duration) -> Result<Self> {
        DateTime::from_unix_duration(unix_duration)?.try_into()
    }

    /// Get the duration of this timestamp since `UNIX_EPOCH`.
    pub fn to_unix_duration(&self) -> Duration {
        self.0.unix_duration()
    }

    /// Instantiate from [`SystemTime`].
    #[cfg(feature = "std")]
    pub fn from_system_time(time: SystemTime) -> Result<Self> {
        DateTime::try_from(time)
            .map_err(|_| Self::TAG.value_error())?
            .try_into()
    }

    /// Convert to [`SystemTime`].
    #[cfg(feature = "std")]
    pub fn to_system_time(&self) -> SystemTime {
        self.0.to_system_time()
    }
}

impl_any_conversions!(UtcTime);

impl<'a> DecodeValue<'a> for UtcTime {
    fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
        if Self::LENGTH != usize::try_from(header.length)? {
            return Err(Self::TAG.value_error());
        }

        let mut bytes = [0u8; Self::LENGTH];
        reader.read_into(&mut bytes)?;

        match bytes {
            // RFC 5280 requires mandatory seconds and Z-normalized time zone
            [year1, year2, mon1, mon2, day1, day2, hour1, hour2, min1, min2, sec1, sec2, b'Z'] => {
                let year = u16::from(datetime::decode_decimal(Self::TAG, year1, year2)?);
                let month = datetime::decode_decimal(Self::TAG, mon1, mon2)?;
                let day = datetime::decode_decimal(Self::TAG, day1, day2)?;
                let hour = datetime::decode_decimal(Self::TAG, hour1, hour2)?;
                let minute = datetime::decode_decimal(Self::TAG, min1, min2)?;
                let second = datetime::decode_decimal(Self::TAG, sec1, sec2)?;

                // RFC 5280 rules for interpreting the year
                let year = if year >= 50 {
                    year.checked_add(1900)
                } else {
                    year.checked_add(2000)
                }
                .ok_or(ErrorKind::DateTime)?;

                DateTime::new(year, month, day, hour, minute, second)
                    .map_err(|_| Self::TAG.value_error())
                    .and_then(|dt| Self::from_unix_duration(dt.unix_duration()))
            }
            _ => Err(Self::TAG.value_error()),
        }
    }
}

impl EncodeValue for UtcTime {
    fn value_len(&self) -> Result<Length> {
        Self::LENGTH.try_into()
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        let year = match self.0.year() {
            y @ 1950..=1999 => y.checked_sub(1900),
            y @ 2000..=2049 => y.checked_sub(2000),
            _ => return Err(Self::TAG.value_error()),
        }
        .and_then(|y| u8::try_from(y).ok())
        .ok_or(ErrorKind::DateTime)?;

        datetime::encode_decimal(writer, Self::TAG, year)?;
        datetime::encode_decimal(writer, Self::TAG, self.0.month())?;
        datetime::encode_decimal(writer, Self::TAG, self.0.day())?;
        datetime::encode_decimal(writer, Self::TAG, self.0.hour())?;
        datetime::encode_decimal(writer, Self::TAG, self.0.minutes())?;
        datetime::encode_decimal(writer, Self::TAG, self.0.seconds())?;
        writer.write_byte(b'Z')
    }
}

impl FixedTag for UtcTime {
    const TAG: Tag = Tag::UtcTime;
}

impl OrdIsValueOrd for UtcTime {}

impl From<&UtcTime> for UtcTime {
    fn from(value: &UtcTime) -> UtcTime {
        *value
    }
}

impl From<UtcTime> for DateTime {
    fn from(utc_time: UtcTime) -> DateTime {
        utc_time.0
    }
}

impl From<&UtcTime> for DateTime {
    fn from(utc_time: &UtcTime) -> DateTime {
        utc_time.0
    }
}

impl TryFrom<DateTime> for UtcTime {
    type Error = Error;

    fn try_from(datetime: DateTime) -> Result<Self> {
        Self::from_date_time(datetime)
    }
}

impl TryFrom<&DateTime> for UtcTime {
    type Error = Error;

    fn try_from(datetime: &DateTime) -> Result<Self> {
        Self::from_date_time(*datetime)
    }
}

#[cfg(feature = "std")]
impl From<UtcTime> for SystemTime {
    fn from(utc_time: UtcTime) -> SystemTime {
        utc_time.to_system_time()
    }
}

// Implement by hand because the derive would create invalid values.
// Use the conversion from DateTime to create a valid value.
// The DateTime type has a way bigger range of valid years than UtcTime,
// so the DateTime year is mapped into a valid range to throw away less inputs.
#[cfg(feature = "arbitrary")]
impl<'a> arbitrary::Arbitrary<'a> for UtcTime {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        const MIN_YEAR: u16 = 1970;
        const VALID_YEAR_COUNT: u16 = UtcTime::MAX_YEAR - MIN_YEAR + 1;
        const AVERAGE_SECONDS_IN_YEAR: u64 = 31_556_952;

        let datetime = DateTime::arbitrary(u)?;
        let year = datetime.year();
        let duration = datetime.unix_duration();

        // Clamp the year into a valid range to not throw away too much input
        let valid_year = (year.saturating_sub(MIN_YEAR))
            .rem_euclid(VALID_YEAR_COUNT)
            .saturating_add(MIN_YEAR);
        let year_to_remove = year.saturating_sub(valid_year);
        let valid_duration = duration
            - Duration::from_secs(
                u64::from(year_to_remove).saturating_mul(AVERAGE_SECONDS_IN_YEAR),
            );

        Self::from_date_time(DateTime::from_unix_duration(valid_duration).expect("supported range"))
            .map_err(|_| arbitrary::Error::IncorrectFormat)
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        DateTime::size_hint(depth)
    }
}

#[cfg(test)]
mod tests {
    use super::UtcTime;
    use crate::{Decode, Encode, SliceWriter};
    use hex_literal::hex;

    #[test]
    fn round_trip_vector() {
        let example_bytes = hex!("17 0d 39 31 30 35 30 36 32 33 34 35 34 30 5a");
        let utc_time = UtcTime::from_der(&example_bytes).unwrap();
        assert_eq!(utc_time.to_unix_duration().as_secs(), 673573540);

        let mut buf = [0u8; 128];
        let mut encoder = SliceWriter::new(&mut buf);
        utc_time.encode(&mut encoder).unwrap();
        assert_eq!(example_bytes, encoder.finish().unwrap());
    }
}
