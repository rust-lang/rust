use std::{
    collections::BTreeMap,
    convert::TryInto as _,
    env, fmt, fs,
    path::{Path, PathBuf},
};

use chrono::{Datelike as _, TimeZone as _, Utc};
use glob::glob;
use regex::Regex;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct Date {
    year: u32,
    month: u32,
}

impl Date {
    fn months_since(self, other: Date) -> Option<u32> {
        let self_chrono = Utc.ymd(self.year.try_into().unwrap(), self.month, 1);
        let other_chrono = Utc.ymd(other.year.try_into().unwrap(), other.month, 1);
        let duration_since = self_chrono.signed_duration_since(other_chrono);
        let months_since = duration_since.num_days() / 30;
        if months_since < 0 {
            None
        } else {
            Some(months_since.try_into().unwrap())
        }
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
        <!--\s*
        [dD]ate:\s*
        (?P<y>\d{4}) # year
        -
        (?P<m>\d{2}) # month
        \s*-->",
    )
    .unwrap()
}

fn collect_dates_from_file(date_regex: &Regex, text: &str) -> Vec<(usize, Date)> {
    let mut line = 1;
    let mut end_of_last_cap = 0;
    date_regex
        .captures_iter(&text)
        .map(|cap| {
            (
                cap.get(0).unwrap().range(),
                Date {
                    year: cap["y"].parse().unwrap(),
                    month: cap["m"].parse().unwrap(),
                },
            )
        })
        .map(|(byte_range, date)| {
            line += text[end_of_last_cap..byte_range.end]
                .chars()
                .filter(|c| *c == '\n')
                .count();
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
    let root_dir = env::args()
        .nth(1)
        .expect("expect root Markdown directory as CLI argument");
    let root_dir_path = Path::new(&root_dir);
    let glob_pat = format!("{}/**/*.md", root_dir);
    let today_chrono = Utc::today();
    let current_month = Date {
        year: today_chrono.year_ce().1,
        month: today_chrono.month(),
    };

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
            current_month = current_month
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
            println!(
                "- [ ] {}",
                path.strip_prefix(&root_dir_path).unwrap().display()
            );
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
        let date1 = Date {
            year: 2020,
            month: 3,
        };
        let date2 = Date {
            year: 2021,
            month: 1,
        };
        assert_eq!(date2.months_since(date1), Some(10));
    }

    #[test]
    fn test_date_regex() {
        let regex = make_date_regex();
        assert!(regex.is_match("foo <!-- date: 2021-01 --> bar"));
    }

    #[test]
    fn test_date_regex_capitalized() {
        let regex = make_date_regex();
        assert!(regex.is_match("foo <!-- Date: 2021-08 --> bar"));
    }

    #[test]
    fn test_collect_dates_from_file() {
        let text = "Test1\n<!-- date: 2021-01 -->\nTest2\nFoo<!-- date: 2021-02 \
                    -->\nTest3\nTest4\nFoo<!-- date: 2021-03 -->Bar\n<!-- date: 2021-04 \
                    -->\nTest5\nTest6\nTest7\n<!-- date: \n\n2021-05 -->\nTest8
        ";
        assert_eq!(
            collect_dates_from_file(&make_date_regex(), text),
            vec![
                (
                    2,
                    Date {
                        year: 2021,
                        month: 1,
                    }
                ),
                (
                    4,
                    Date {
                        year: 2021,
                        month: 2,
                    }
                ),
                (
                    7,
                    Date {
                        year: 2021,
                        month: 3,
                    }
                ),
                (
                    8,
                    Date {
                        year: 2021,
                        month: 4,
                    }
                ),
                (
                    14,
                    Date {
                        year: 2021,
                        month: 5,
                    }
                ),
            ]
        );
    }
}
