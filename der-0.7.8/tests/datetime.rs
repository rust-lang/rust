//! Tests for the [`DateTime`] type.

use der::{asn1::UtcTime, DateTime, Decode, Encode};
use proptest::prelude::*;

proptest! {
    #[test]
    fn roundtrip_datetime(
        year in 1970u16..=9999,
        month in 1u8..=12,
        day in 1u8..=31,
        hour in 0u8..=23,
        min in 0u8..=59,
        sec in 0u8..=59,
    ) {
        let datetime1 = make_datetime(year, month, day, hour, min, sec);
        let datetime2 = DateTime::from_unix_duration(datetime1.unix_duration()).unwrap();
        prop_assert_eq!(datetime1, datetime2);
    }

    #[test]
    fn roundtrip_utctime(
        year in 1970u16..=2049,
        month in 1u8..=12,
        day in 1u8..=31,
        hour in 0u8..=23,
        min in 0u8..=59,
        sec in 0u8..=59,
    ) {
        let datetime = make_datetime(year, month, day, hour, min, sec);
        let utc_time1 = UtcTime::try_from(datetime).unwrap();

        let mut buf = [0u8; 128];
        let mut encoder = der::SliceWriter::new(&mut buf);
        utc_time1.encode(&mut encoder).unwrap();
        let der_bytes = encoder.finish().unwrap();

        let utc_time2 = UtcTime::from_der(der_bytes).unwrap();
        prop_assert_eq!(utc_time1, utc_time2);
    }
}

fn make_datetime(year: u16, month: u8, day: u8, hour: u8, min: u8, sec: u8) -> DateTime {
    let max_day = if month == 2 {
        let is_leap_year = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);

        if is_leap_year {
            29
        } else {
            28
        }
    } else {
        30
    };

    let day = if day > max_day { max_day } else { day };

    DateTime::new(year, month, day, hour, min, sec).unwrap_or_else(|e| {
        panic!(
            "invalid DateTime: {:02}-{:02}-{:02}T{:02}:{:02}:{:02}: {}",
            year, month, day, hour, min, sec, e
        );
    })
}
