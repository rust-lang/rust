use fmt;
use time::Duration;


// Number of seconds in a day is a constant.
// We do not support leap seconds here.
const SECONDS_IN_DAY: u64 = 86400;


// Gregorian calendar has 400 years cycles, this is a procedure
// for computing if a year is a leap year.
fn is_leap_year(year: i64) -> bool {
    if year % 4 != 0 {
        false
    } else if year % 100 != 0 {
        true
    } else if year % 400 != 0 {
        false
    } else {
        true
    }
}

fn days_in_year(year: i64) -> u32 {
    if is_leap_year(year) {
        366
    } else {
        365
    }
}


// Number of leap years among 400 consecutive years.
const CYCLE_LEAP_YEARS: u32 = 400 / 4 - 400 / 100 + 400 / 400;
// Number of days in 400 years cycle.
const CYCLE_DAYS: u32 = 400 * 365 + CYCLE_LEAP_YEARS;
const CYCLE_SECONDS: u64 = CYCLE_DAYS as u64 * SECONDS_IN_DAY;


// Number of seconds between 1 Jan 1970 and 1 Jan 2000.
// Check with:
// `TZ=UTC gdate --rfc-3339=seconds --date @946684800`
const YEARS_1970_2000_SECONDS: u64 = 946684800;
// Number of seconds between 1 Jan 1600 and 1 Jan 1970.
const YEARS_1600_1970_SECONDS: u64 = CYCLE_SECONDS - YEARS_1970_2000_SECONDS;


// For each year in the cycle, number of leap years before in the cycle.
static YEAR_DELTAS: [u8; 401] = [
     0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  4,  5,  5,  5,
     5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  8,  9,  9,  9,  9, 10, 10, 10,
    10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15,
    15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20,
    20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, // 100
    25, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29,
    29, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34,
    34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39,
    39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 44,
    44, 45, 45, 45, 45, 46, 46, 46, 46, 47, 47, 47, 47, 48, 48, 48, 48, 49, 49, 49, // 200
    49, 49, 49, 49, 49, 50, 50, 50, 50, 51, 51, 51, 51, 52, 52, 52, 52, 53, 53, 53,
    53, 54, 54, 54, 54, 55, 55, 55, 55, 56, 56, 56, 56, 57, 57, 57, 57, 58, 58, 58,
    58, 59, 59, 59, 59, 60, 60, 60, 60, 61, 61, 61, 61, 62, 62, 62, 62, 63, 63, 63,
    63, 64, 64, 64, 64, 65, 65, 65, 65, 66, 66, 66, 66, 67, 67, 67, 67, 68, 68, 68,
    68, 69, 69, 69, 69, 70, 70, 70, 70, 71, 71, 71, 71, 72, 72, 72, 72, 73, 73, 73, // 300
    73, 73, 73, 73, 73, 74, 74, 74, 74, 75, 75, 75, 75, 76, 76, 76, 76, 77, 77, 77,
    77, 78, 78, 78, 78, 79, 79, 79, 79, 80, 80, 80, 80, 81, 81, 81, 81, 82, 82, 82,
    82, 83, 83, 83, 83, 84, 84, 84, 84, 85, 85, 85, 85, 86, 86, 86, 86, 87, 87, 87,
    87, 88, 88, 88, 88, 89, 89, 89, 89, 90, 90, 90, 90, 91, 91, 91, 91, 92, 92, 92,
    92, 93, 93, 93, 93, 94, 94, 94, 94, 95, 95, 95, 95, 96, 96, 96, 96, 97, 97, 97, 97,
];


/// UTC time
pub struct TmUtc {
    /// Year
    year: i64,
    /// 1..=12
    month: u32,
    /// 1-based day of month
    day: u32,
    /// 0..=23
    hour: u32,
    /// 0..=59
    minute: u32,
    /// 0..=59; no leap seconds
    second: u32,
    /// 0..=999_999_999
    nanos: u32,
}

impl TmUtc {

    fn day_of_cycle_to_year_day_of_year(day_of_cycle: u32) -> (i64, u32) {
        debug_assert!(day_of_cycle < CYCLE_DAYS);

        let mut year_mod_400 = (day_of_cycle / 365) as i64;
        let mut day_or_year = (day_of_cycle % 365) as u32;

        let delta = YEAR_DELTAS[year_mod_400 as usize] as u32;
        if day_or_year < delta {
            year_mod_400 -= 1;
            day_or_year += 365 - YEAR_DELTAS[year_mod_400 as usize] as u32;
        } else {
            day_or_year -= delta;
        }

        (year_mod_400, day_or_year)
    }

    // Convert seconds of the day of hour, minute and second
    fn seconds_of_day_to_h_m_s(seconds: u32) -> (u32, u32, u32) {
        debug_assert!(seconds < SECONDS_IN_DAY as u32);

        let hour = seconds / 3600;
        let minute = seconds % 3600 / 60;
        let second = seconds % 60;

        (hour, minute, second)
    }

    // Convert day of year (0-based) to month and day
    fn day_of_year_to_month_day(year: i64, day_of_year: u32) -> (u32, u32) {
        debug_assert!(day_of_year < days_in_year(year));

        let days_in_months = if is_leap_year(year) {
            &[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        } else {
            &[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        };

        let mut rem_days = day_of_year;
        let mut month = 1;
        while rem_days >= days_in_months[month - 1] {
            rem_days -= days_in_months[month - 1];
            month += 1;
        }

        debug_assert!(rem_days + 1 <= days_in_months[month - 1]);

        (month as u32, rem_days + 1)
    }

    // Construct from duration added to cycle start year
    fn from_cycle_start_add_duration(mut cycle_start: i64, add: Duration) -> TmUtc {
        debug_assert!(cycle_start % 400 == 0);

        // Split duration to days and duration within day

        let days = add.as_secs() / SECONDS_IN_DAY;
        let duration_of_day = add - Duration::from_secs(days * SECONDS_IN_DAY);

        let cycles = days / CYCLE_DAYS as u64;
        cycle_start += cycles as i64 * 400;
        let day_of_cycle = days % CYCLE_DAYS as u64;

        let (year_mod_400, day_of_year) =
            TmUtc::day_of_cycle_to_year_day_of_year(day_of_cycle as u32);

        let (year,) = (cycle_start + year_mod_400,);
        let (month, day) = TmUtc::day_of_year_to_month_day(year, day_of_year);
        let (hour, minute, second) =
            TmUtc::seconds_of_day_to_h_m_s(duration_of_day.as_secs() as u32);

        TmUtc {
            year,
            month,
            day,
            hour,
            minute,
            second,
            nanos: duration_of_day.subsec_nanos(),
        }
    }

    // Construct from duration added to epoch
    pub fn from_epoch_add(add: Duration) -> TmUtc {
        // Special case to prevent integer overflow in duration addition
        if add > Duration::from_secs(YEARS_1970_2000_SECONDS) {
            TmUtc::from_cycle_start_add_duration(
                2000, add - Duration::from_secs(YEARS_1970_2000_SECONDS))
        } else {
            TmUtc::from_cycle_start_add_duration(
                1600, add + Duration::from_secs(YEARS_1600_1970_SECONDS))
        }
    }

    // Construct from duration subtracted from epoch
    pub fn from_epoch_sub(mut sub: Duration) -> TmUtc {

        // Make `sub` less than 400 years

        // Number of full leap cycles in `sub`
        let leap_cycles = sub.as_secs() / CYCLE_SECONDS;
        let year = 1970 - (leap_cycles * 400) as i64;
        sub -= Duration::from_secs(leap_cycles * CYCLE_SECONDS);

        // Convert subtraction to addition
        // and align year to a cycle start

        // `year` is still 370 years ahead of cycle start.
        debug_assert!((year % 400 + 400) % 400 == 370);

        // Subtract 370 years to get to cycle start
        // and additional 400 year to prevent integer underflow in duration subtraction
        let cycle_start = year - 400 - 370;
        let add_from_cycle_start =
            Duration::from_secs(CYCLE_SECONDS + YEARS_1600_1970_SECONDS) - sub;

        debug_assert!(cycle_start % 400 == 0);

        return TmUtc::from_cycle_start_add_duration(cycle_start, add_from_cycle_start);
    }

    pub fn fmt_iso_8601(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.year > 9999 {
            // ISO-8601 extended format requires leading plus
            write!(f, "+{}", self.year)?;
        } else if self.year < 0 {
            // At least 4 digits
            write!(f, "{:05}", self.year)?;
        } else {
            write!(f, "{:04}", self.year)?;
        }

        write!(f, "-{:02}-{:02}T{:02}:{:02}:{:02}",
            self.month, self.day, self.hour, self.minute, self.second)?;

        // if width is not specified, print nanoseconds
        let subsec_digits = f.precision().unwrap_or(9);
        if subsec_digits != 0 {
            let width = if subsec_digits > 9 { 9 } else { subsec_digits };

            // "Truncated" nanonseconds.
            let mut subsec = self.nanos;

            // Performs 8 iterations when width=1,
            // but that's probably not a issue compared to other computations.
            for _ in width..9 {
                subsec /= 10;
            }

            write!(f, ".{:0width$}", subsec, width=width as usize)?;

            // Adding more than 9 digits is meaningless,
            // but if user requests it, we should print zeros.
            if subsec_digits > 9 {
                write!(f, "{:0n$}", 0, n=subsec_digits - 9)?;
            }
        }

        write!(f, "Z")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use time::Duration;
    use fmt;
    use u64;

    struct TmUtcDisplayIso8601(TmUtc);

    impl fmt::Display for TmUtcDisplayIso8601 {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            self.0.fmt_iso_8601(f)
        }
    }

    #[test]
    fn test_fmt() {
        fn test_impl(expected: &str, secs: i64, nanos: i32, subsec_digits: u32) {
            if nanos != 0 {
                assert_eq!(secs >= 0, nanos >= 0);
            }

            let time = TmUtcDisplayIso8601(if secs >= 0 {
                TmUtc::from_epoch_add(Duration::new(secs as u64, nanos as u32))
            } else {
                TmUtc::from_epoch_sub(Duration::new(-secs as u64, -nanos as u32))
            });

            assert_eq!(expected, format!("{:.digits$}", time, digits=subsec_digits as usize));
        }

        // Tests can be validated with with GNU date:
        // `TZ=UTC gdate --date @1535585179 --iso-8601=seconds`

        test_impl("1970-01-01T00:00:00Z", 0, 0, 0);
        test_impl("2018-08-29T23:26:19Z", 1535585179, 0, 0);
        test_impl("2018-08-29T23:26:19.123Z", 1535585179, 123456789, 3);
        test_impl("1646-04-01T03:45:44Z", -10216613656, 0, 0);
        // Last day of year in regular and leap year
        test_impl("2018-12-31T00:00:00Z", 1546214400, 0, 0);
        test_impl("2020-12-31T00:00:00Z", 1609372800, 0, 0);
        // More than 9 digits in precision
        test_impl("1970-01-01T00:00:00.000000001000Z", 0, 1, 12);
        // Large years
        test_impl("5138-11-16T09:46:40Z", 100000000000, 0, 0);
        // Leading zero
        test_impl("0000-12-31T00:00:00Z", -62135683200, 0, 0);
        // Minus zero
        test_impl("-0003-10-30T14:13:20Z", -62235683200, 0, 0);
        // More than 4 digits
        test_impl("+33658-09-27T01:46:41Z", 1000000000001, 0, 0);
        // Largest value GNU date can handle
        test_impl("+2147485547-12-31T23:59:59Z", 67768036191676799, 0, 0);
        // Negative dates
        test_impl("1969-12-31T23:59:59Z", -1, 0, 0);
        test_impl("1969-12-31T23:59:00Z", -60, 0, 0);
        test_impl("1969-12-31T23:59:58.900Z", -1, -100_000_000, 3);
        test_impl("1966-10-31T14:13:20Z", -100000000, 0, 0);
        test_impl("-29719-04-05T22:13:19Z", -1000000000001, 0, 0);
        // Smallest value GNU date can handle
        test_impl("-2147481748-01-01T00:00:00Z", -67768040609740800, 0, 0);
    }

    #[test]
    fn test_fmt_max_duration() {
        let duration = Duration::new(u64::max_value(), 999_999_999);
        // Simply check that there are no integer overflows.
        // I didn't check that resulting strings are correct.
        assert_eq!("+584554051223-11-09T07:00:15.999999999Z",
            format!("{}", TmUtcDisplayIso8601(TmUtc::from_epoch_add(duration))));
        assert_eq!("-584554047284-02-23T16:59:44.000000001Z",
            format!("{}", TmUtcDisplayIso8601(TmUtc::from_epoch_sub(duration))));
    }
}
