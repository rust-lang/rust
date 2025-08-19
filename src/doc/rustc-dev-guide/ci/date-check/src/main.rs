use std::collections::BTreeMap;
use std::convert::TryInto as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::{env, fmt, fs, process};

use chrono::{Datelike as _, Month, TimeZone as _, Utc};
use glob::glob;
use regex::Regex;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Date {
    year: u32,
    month: u32,
}

impl Date {
    fn months_since(self, other: Date) -> Option<u32> {
        let self_chrono =
            Utc.with_ymd_and_hms(self.year.try_into().unwrap(), self.month, 1, 0, 0, 0).unwrap();
        let other_chrono =
            Utc.with_ymd_and_hms(other.year.try_into().unwrap(), other.month, 1, 0, 0, 0).unwrap();
        let duration_since = self_chrono.signed_duration_since(other_chrono);
        let months_since = duration_since.num_days() / 30;
        if months_since < 0 { None } else { Some(months_since.try_into().unwrap()) }
    }
}

impl fmt::Display for Date {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:04}-{:02}", self.year, self.month)
    }
}

fn make_date_regex() -> Regex {
    Regex::new(
        r"(?x) # insignificant whitespace mode
        (<!--\s*
          date-check:\s*
          (?P<m1>[[:alpha:]]+)\s+
          (?P<y1>\d{4})\s*-->
        )
        |
        (<!--\s*
          date-check\s*-->\s+
          (?P<m2>[[:alpha:]]+)\s+
          (?P<y2>\d{4})\b
        )
    ",
    )
    .unwrap()
}

fn collect_dates_from_file(date_regex: &Regex, text: &str) -> Vec<(usize, Date)> {
    let mut line = 1;
    let mut end_of_last_cap = 0;
    date_regex
        .captures_iter(text)
        .filter_map(|cap| {
            if let (Some(month), Some(year), None, None) | (None, None, Some(month), Some(year)) =
                (cap.name("m1"), cap.name("y1"), cap.name("m2"), cap.name("y2"))
            {
                let year = year.as_str().parse().expect("year");
                let month = Month::from_str(month.as_str()).expect("month").number_from_month();
                Some((cap.get(0).expect("all").range(), Date { year, month }))
            } else {
                None
            }
        })
        .map(|(byte_range, date)| {
            line += text[end_of_last_cap..byte_range.end].chars().filter(|c| *c == '\n').count();
            end_of_last_cap = byte_range.end;
            (line, date)
        })
        .collect()
}

fn collect_dates(paths: impl Iterator<Item = PathBuf>) -> BTreeMap<PathBuf, Vec<(usize, Date)>> {
    let date_regex = make_date_regex();
    let mut data = BTreeMap::new();
    for path in paths {
        let text = fs::read_to_string(&path).unwrap();
        let dates = collect_dates_from_file(&date_regex, &text);
        if !dates.is_empty() {
            data.insert(path, dates);
        }
    }
    data
}

fn filter_dates(
    current_month: Date,
    min_months_since: u32,
    dates_by_file: impl Iterator<Item = (PathBuf, Vec<(usize, Date)>)>,
) -> impl Iterator<Item = (PathBuf, Vec<(usize, Date)>)> {
    dates_by_file
        .map(move |(path, dates)| {
            (
                path,
                dates
                    .into_iter()
                    .filter(|(_, date)| {
                        current_month
                            .months_since(*date)
                            .expect("found date that is after current month")
                            >= min_months_since
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .filter(|(_, dates)| !dates.is_empty())
}

fn main() {
    let mut args = env::args();
    if args.len() == 1 {
        eprintln!("error: expected root of Markdown directory as CLI argument");
        process::exit(1);
    }
    let root_dir = args.nth(1).unwrap();
    let root_dir_path = Path::new(&root_dir);
    let glob_pat = format!("{}/**/*.md", root_dir);
    let today_chrono = Utc::now().date_naive();
    let current_month = Date { year: today_chrono.year_ce().1, month: today_chrono.month() };

    let dates_by_file = collect_dates(glob(&glob_pat).unwrap().map(Result::unwrap));
    let dates_by_file: BTreeMap<_, _> =
        filter_dates(current_month, 6, dates_by_file.into_iter()).collect();

    if dates_by_file.is_empty() {
        println!("empty");
    } else {
        println!("Date Reference Triage for {}", current_month);
        println!("## Procedure");
        println!();
        println!(
            "Each of these dates should be checked to see if the docs they annotate are \
             up-to-date. Each date should be updated (in the Markdown file where it appears) to \
             use the current month ({current_month}), or removed if the docs it annotates are not \
             expected to fall out of date quickly.",
            current_month = today_chrono.format("%B %Y"),
        );
        println!();
        println!(
            "Please check off each date once a PR to update it (and, if applicable, its \
             surrounding docs) has been merged. Please also mention that you are working on a \
             particular set of dates so duplicate work is avoided."
        );
        println!();
        println!("Finally, once all the dates have been updated, please close this issue.");
        println!();
        println!("## Dates");
        println!();

        for (path, dates) in dates_by_file {
            println!("- {}", path.strip_prefix(&root_dir_path).unwrap_or(&path).display(),);
            for (line, date) in dates {
                println!("  - [ ] line {}: {}", line, date);
            }
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_months_since() {
        let date1 = Date { year: 2020, month: 3 };
        let date2 = Date { year: 2021, month: 1 };
        assert_eq!(date2.months_since(date1), Some(10));
    }

    #[test]
    fn test_date_regex() {
        let regex = &make_date_regex();
        assert!(regex.is_match("<!-- date-check: jan 2021 -->"));
        assert!(regex.is_match("<!-- date-check: january 2021 -->"));
        assert!(regex.is_match("<!-- date-check: Jan 2021 -->"));
        assert!(regex.is_match("<!-- date-check: January 2021 -->"));
        assert!(regex.is_match("<!-- date-check --> jan 2021"));
        assert!(regex.is_match("<!-- date-check --> january 2021"));
        assert!(regex.is_match("<!-- date-check --> Jan 2021"));
        assert!(regex.is_match("<!-- date-check --> January 2021"));

        assert!(regex.is_match("<!-- date-check --> jan 2021 "));
        assert!(regex.is_match("<!-- date-check --> jan 2021."));
    }

    #[test]
    fn test_date_regex_fail() {
        let regexes = &make_date_regex();
        assert!(!regexes.is_match("<!-- date-check: jan 221 -->"));
        assert!(!regexes.is_match("<!-- date-check: jan 20221 -->"));
        assert!(!regexes.is_match("<!-- date-check: 01 2021 -->"));
        assert!(!regexes.is_match("<!-- date-check --> jan 221"));
        assert!(!regexes.is_match("<!-- date-check --> jan 20222"));
        assert!(!regexes.is_match("<!-- date-check --> 01 2021"));
    }

    #[test]
    fn test_collect_dates_from_file() {
        let text = r"
Test1
<!-- date-check: jan 2021 -->
Test2
Foo<!-- date-check: february 2021
-->
Test3
Test4
Foo<!-- date-check: Mar 2021 -->Bar
<!-- date-check:April 2021
-->
Test5
Test6
Test7
<!-- date-check:

may 2021 -->
Test8
Test1
<!-- date-check -->  jan 2021
Test2
Foo<!-- date-check
--> february 2021
Test3
Test4
Foo<!-- date-check -->  mar 2021 Bar
<!-- date-check
--> apr 2021
Test5
Test6
Test7
<!-- date-check

 --> may 2021
Test8
 <!--
   date-check
 --> june 2021.
        ";
        assert_eq!(
            collect_dates_from_file(&make_date_regex(), text),
            vec![
                (3, Date { year: 2021, month: 1 }),
                (6, Date { year: 2021, month: 2 }),
                (9, Date { year: 2021, month: 3 }),
                (11, Date { year: 2021, month: 4 }),
                (17, Date { year: 2021, month: 5 }),
                (20, Date { year: 2021, month: 1 }),
                (23, Date { year: 2021, month: 2 }),
                (26, Date { year: 2021, month: 3 }),
                (28, Date { year: 2021, month: 4 }),
                (34, Date { year: 2021, month: 5 }),
                (38, Date { year: 2021, month: 6 }),
            ],
        );
    }
}
