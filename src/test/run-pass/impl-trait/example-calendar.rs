// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(conservative_impl_trait, fn_traits, step_trait, unboxed_closures)]

//! Derived from: <https://raw.githubusercontent.com/quickfur/dcal/master/dcal.d>.
//!
//! Originally converted to Rust by [Daniel Keep](https://github.com/DanielKeep).

use std::fmt::Write;
use std::mem;

/// Date representation.
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct NaiveDate(i32, u32, u32);

impl NaiveDate {
    pub fn from_ymd(y: i32, m: u32, d: u32) -> NaiveDate {
        assert!(1 <= m && m <= 12, "m = {:?}", m);
        assert!(1 <= d && d <= NaiveDate(y, m, 1).days_in_month(), "d = {:?}", d);
        NaiveDate(y, m, d)
    }

    pub fn year(&self) -> i32 {
        self.0
    }

    pub fn month(&self) -> u32 {
        self.1
    }

    pub fn day(&self) -> u32 {
        self.2
    }

    pub fn succ(&self) -> NaiveDate {
        let (mut y, mut m, mut d, n) = (
            self.year(), self.month(), self.day()+1, self.days_in_month());
        if d > n {
            d = 1;
            m += 1;
        }
        if m > 12 {
            m = 1;
            y += 1;
        }
        NaiveDate::from_ymd(y, m, d)
    }

    pub fn weekday(&self) -> Weekday {
        use Weekday::*;

        // 0 = Sunday
        let year = self.year();
        let dow_jan_1 = (year*365 + ((year-1) / 4) - ((year-1) / 100) + ((year-1) / 400)) % 7;
        let dow = (dow_jan_1 + (self.day_of_year() as i32 - 1)) % 7;
        [Sun, Mon, Tue, Wed, Thu, Fri, Sat][dow as usize]
    }

    pub fn isoweekdate(&self) -> (i32, u32, Weekday) {
        let first_dow_mon_0 = self.year_first_day_of_week().num_days_from_monday();

        // Work out this date's DOtY and week number, not including year adjustment.
        let doy_0 = self.day_of_year() - 1;
        let mut week_mon_0: i32 = ((first_dow_mon_0 + doy_0) / 7) as i32;

        if self.first_week_in_prev_year() {
            week_mon_0 -= 1;
        }

        let weeks_in_year = self.last_week_number();

        // Work out the final result.
        // If the week is -1 or >= weeks_in_year, we will need to adjust the year.
        let year = self.year();
        let wd = self.weekday();

        if week_mon_0 < 0 {
            (year - 1, NaiveDate::from_ymd(year - 1, 1, 1).last_week_number(), wd)
        } else if week_mon_0 >= weeks_in_year as i32 {
            (year + 1, (week_mon_0 + 1 - weeks_in_year as i32) as u32, wd)
        } else {
            (year, (week_mon_0 + 1) as u32, wd)
        }
    }

    fn first_week_in_prev_year(&self) -> bool {
        let first_dow_mon_0 = self.year_first_day_of_week().num_days_from_monday();

        // Any day in the year *before* the first Monday of that year
        // is considered to be in the last week of the previous year,
        // assuming the first week has *less* than four days in it.
        // Adjust the week appropriately.
        ((7 - first_dow_mon_0) % 7) < 4
    }

    fn year_first_day_of_week(&self) -> Weekday {
        NaiveDate::from_ymd(self.year(), 1, 1).weekday()
    }

    fn weeks_in_year(&self) -> u32 {
        let days_in_last_week = self.year_first_day_of_week().num_days_from_monday() + 1;
        if days_in_last_week >= 4 { 53 } else { 52 }
    }

    fn last_week_number(&self) -> u32 {
        let wiy = self.weeks_in_year();
        if self.first_week_in_prev_year() { wiy - 1 } else { wiy }
    }

    fn day_of_year(&self) -> u32 {
        (1..self.1).map(|m| NaiveDate::from_ymd(self.year(), m, 1).days_in_month())
            .fold(0, |a,b| a+b) + self.day()
    }

    fn is_leap_year(&self) -> bool {
        let year = self.year();
        if year % 4 != 0 {
            return false
        } else if year % 100 != 0 {
            return true
        } else if year % 400 != 0 {
            return false
        } else {
            return true
        }
    }

    fn days_in_month(&self) -> u32 {
        match self.month() {
            /* Jan */ 1 => 31,
            /* Feb */ 2 => if self.is_leap_year() { 29 } else { 28 },
            /* Mar */ 3 => 31,
            /* Apr */ 4 => 30,
            /* May */ 5 => 31,
            /* Jun */ 6 => 30,
            /* Jul */ 7 => 31,
            /* Aug */ 8 => 31,
            /* Sep */ 9 => 30,
            /* Oct */ 10 => 31,
            /* Nov */ 11 => 30,
            /* Dec */ 12 => 31,
            _ => unreachable!()
        }
    }
}

impl<'a, 'b> std::ops::Add<&'b NaiveDate> for &'a NaiveDate {
    type Output = NaiveDate;

    fn add(self, other: &'b NaiveDate) -> NaiveDate {
        assert_eq!(*other, NaiveDate(0, 0, 1));
        self.succ()
    }
}

impl std::iter::Step for NaiveDate {
    fn step(&self, by: &Self) -> Option<Self> {
        Some(self + by)
    }

    fn steps_between(_: &Self, _: &Self, _: &Self) -> Option<usize> {
        unimplemented!()
    }

    fn steps_between_by_one(_: &Self, _: &Self) -> Option<usize> {
        unimplemented!()
    }

    fn is_negative(&self) -> bool {
        false
    }

    fn replace_one(&mut self) -> Self {
        mem::replace(self, NaiveDate(0, 0, 1))
    }

    fn replace_zero(&mut self) -> Self {
        mem::replace(self, NaiveDate(0, 0, 0))
    }

    fn add_one(&self) -> Self {
        self.succ()
    }

    fn sub_one(&self) -> Self {
        unimplemented!()
    }
}

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Weekday {
    Mon,
    Tue,
    Wed,
    Thu,
    Fri,
    Sat,
    Sun,
}

impl Weekday {
    pub fn num_days_from_monday(&self) -> u32 {
        use Weekday::*;
        match *self {
            Mon => 0,
            Tue => 1,
            Wed => 2,
            Thu => 3,
            Fri => 4,
            Sat => 5,
            Sun => 6,
        }
    }

    pub fn num_days_from_sunday(&self) -> u32 {
        use Weekday::*;
        match *self {
            Sun => 0,
            Mon => 1,
            Tue => 2,
            Wed => 3,
            Thu => 4,
            Fri => 5,
            Sat => 6,
        }
    }
}

/// Wrapper for zero-sized closures.
// HACK(eddyb) Only needed because closures can't implement Copy.
struct Fn0<F>(std::marker::PhantomData<F>);

impl<F> Copy for Fn0<F> {}
impl<F> Clone for Fn0<F> {
    fn clone(&self) -> Self { *self }
}

impl<F: FnOnce<A>, A> FnOnce<A> for Fn0<F> {
    type Output = F::Output;

    extern "rust-call" fn call_once(self, args: A) -> Self::Output {
        let f = unsafe { std::mem::uninitialized::<F>() };
        f.call_once(args)
    }
}

impl<F: FnMut<A>, A> FnMut<A> for Fn0<F> {
    extern "rust-call" fn call_mut(&mut self, args: A) -> Self::Output {
        let mut f = unsafe { std::mem::uninitialized::<F>() };
        f.call_mut(args)
    }
}

trait AsFn0<A>: Sized {
    fn copyable(self) -> Fn0<Self>;
}

impl<F: FnMut<A>, A> AsFn0<A> for F {
    fn copyable(self) -> Fn0<Self> {
        assert_eq!(std::mem::size_of::<F>(), 0);
        Fn0(std::marker::PhantomData)
    }
}

/// GroupBy implementation.
struct GroupBy<It: Iterator, F> {
    it: std::iter::Peekable<It>,
    f: F,
}

impl<It, F> Clone for GroupBy<It, F>
where It: Iterator + Clone, It::Item: Clone, F: Clone {
    fn clone(&self) -> GroupBy<It, F> {
        GroupBy {
            it: self.it.clone(),
            f: self.f.clone()
        }
    }
}

impl<'a, G, It: 'a, F: 'a> Iterator for GroupBy<It, F>
where It: Iterator + Clone,
      It::Item: Clone,
      F: Clone + FnMut(&It::Item) -> G,
      G: Eq + Clone
{
    type Item = (G, InGroup<std::iter::Peekable<It>, F, G>);

    fn next(&mut self) -> Option<Self::Item> {
        self.it.peek().map(&mut self.f).map(|key| {
            let start = self.it.clone();
            while let Some(k) = self.it.peek().map(&mut self.f) {
                if key != k {
                    break;
                }
                self.it.next();
            }

            (key.clone(), InGroup {
                it: start,
                f: self.f.clone(),
                g: key
            })
        })
    }
}

#[derive(Copy, Clone)]
struct InGroup<It, F, G> {
    it: It,
    f: F,
    g: G
}

impl<It: Iterator, F: FnMut(&It::Item) -> G, G: Eq> Iterator for InGroup<It, F, G> {
    type Item = It::Item;

    fn next(&mut self) -> Option<It::Item> {
        self.it.next().and_then(|x| {
            if (self.f)(&x) == self.g { Some(x) } else { None }
        })
    }
}

trait IteratorExt: Iterator + Sized {
    fn group_by<G, F>(self, f: F) -> GroupBy<Self, Fn0<F>>
    where F: FnMut(&Self::Item) -> G,
          G: Eq
    {
        GroupBy {
            it: self.peekable(),
            f: f.copyable(),
        }
    }

    fn join(mut self, sep: &str) -> String
    where Self::Item: std::fmt::Display {
        let mut s = String::new();
        if let Some(e) = self.next() {
            write!(s, "{}", e);
            for e in self {
                s.push_str(sep);
                write!(s, "{}", e);
            }
        }
        s
    }

    // HACK(eddyb) Only needed because `impl Trait` can't be
    // used with trait methods: `.foo()` becomes `.__(foo)`.
    fn __<F, R>(self, f: F) -> R
    where F: FnOnce(Self) -> R {
        f(self)
    }
}

impl<It> IteratorExt for It where It: Iterator {}

///
/// Generates an iterator that yields exactly n spaces.
///
fn spaces(n: usize) -> std::iter::Take<std::iter::Repeat<char>> {
    std::iter::repeat(' ').take(n)
}

fn test_spaces() {
    assert_eq!(spaces(0).collect::<String>(), "");
    assert_eq!(spaces(10).collect::<String>(), "          ")
}

///
/// Returns an iterator of dates in a given year.
///
fn dates_in_year(year: i32) -> impl Iterator<Item=NaiveDate>+Clone {
    InGroup {
        it: NaiveDate::from_ymd(year, 1, 1)..,
        f: (|d: &NaiveDate| d.year()).copyable(),
        g: year
    }
}

fn test_dates_in_year() {
    {
        let mut dates = dates_in_year(2013);
        assert_eq!(dates.next(), Some(NaiveDate::from_ymd(2013, 1, 1)));

        // Check increment
        assert_eq!(dates.next(), Some(NaiveDate::from_ymd(2013, 1, 2)));

        // Check monthly rollover
        for _ in 3..31 {
            assert!(dates.next() != None);
        }

        assert_eq!(dates.next(), Some(NaiveDate::from_ymd(2013, 1, 31)));
        assert_eq!(dates.next(), Some(NaiveDate::from_ymd(2013, 2, 1)));
    }

    {
        // Check length of year
        let mut dates = dates_in_year(2013);
        for _ in 0..365 {
            assert!(dates.next() != None);
        }
        assert_eq!(dates.next(), None);
    }

    {
        // Check length of leap year
        let mut dates = dates_in_year(1984);
        for _ in 0..366 {
            assert!(dates.next() != None);
        }
        assert_eq!(dates.next(), None);
    }
}

///
/// Convenience trait for verifying that a given type iterates over
/// `NaiveDate`s.
///
trait DateIterator: Iterator<Item=NaiveDate> + Clone {}
impl<It> DateIterator for It where It: Iterator<Item=NaiveDate> + Clone {}

fn test_group_by() {
    let input = [
        [1, 1],
        [1, 1],
        [1, 2],
        [2, 2],
        [2, 3],
        [2, 3],
        [3, 3]
    ];

    let by_x = input.iter().cloned().group_by(|a| a[0]);
    let expected_1: &[&[[i32; 2]]] = &[
        &[[1, 1], [1, 1], [1, 2]],
        &[[2, 2], [2, 3], [2, 3]],
        &[[3, 3]]
    ];
    for ((_, a), b) in by_x.zip(expected_1.iter().cloned()) {
        assert_eq!(&a.collect::<Vec<_>>()[..], b);
    }

    let by_y = input.iter().cloned().group_by(|a| a[1]);
    let expected_2: &[&[[i32; 2]]] = &[
        &[[1, 1], [1, 1]],
        &[[1, 2], [2, 2]],
        &[[2, 3], [2, 3], [3, 3]]
    ];
    for ((_, a), b) in by_y.zip(expected_2.iter().cloned()) {
        assert_eq!(&a.collect::<Vec<_>>()[..], b);
    }
}

///
/// Groups an iterator of dates by month.
///
fn by_month<It>(it: It)
                ->  impl Iterator<Item=(u32, impl Iterator<Item=NaiveDate> + Clone)> + Clone
where It: Iterator<Item=NaiveDate> + Clone {
    it.group_by(|d| d.month())
}

fn test_by_month() {
    let mut months = dates_in_year(2013).__(by_month);
    for (month, (_, mut date)) in (1..13).zip(&mut months) {
        assert_eq!(date.nth(0).unwrap(), NaiveDate::from_ymd(2013, month, 1));
    }
    assert!(months.next().is_none());
}

///
/// Groups an iterator of dates by week.
///
fn by_week<It>(it: It)
               -> impl Iterator<Item=(u32, impl DateIterator)> + Clone
where It: DateIterator {
    // We go forward one day because `isoweekdate` considers the week to start on a Monday.
    it.group_by(|d| d.succ().isoweekdate().1)
}

fn test_isoweekdate() {
    fn weeks_uniq(year: i32) -> Vec<((i32, u32), u32)> {
        let mut weeks = dates_in_year(year).map(|d| d.isoweekdate())
            .map(|(y,w,_)| (y,w));
        let mut result = vec![];
        let mut accum = (weeks.next().unwrap(), 1);
        for yw in weeks {
            if accum.0 == yw {
                accum.1 += 1;
            } else {
                result.push(accum);
                accum = (yw, 1);
            }
        }
        result.push(accum);
        result
    }

    let wu_1984 = weeks_uniq(1984);
    assert_eq!(&wu_1984[..2], &[((1983, 52), 1), ((1984, 1), 7)]);
    assert_eq!(&wu_1984[wu_1984.len()-2..], &[((1984, 52), 7), ((1985, 1), 1)]);

    let wu_2013 = weeks_uniq(2013);
    assert_eq!(&wu_2013[..2], &[((2013, 1), 6), ((2013, 2), 7)]);
    assert_eq!(&wu_2013[wu_2013.len()-2..], &[((2013, 52), 7), ((2014, 1), 2)]);

    let wu_2015 = weeks_uniq(2015);
    assert_eq!(&wu_2015[..2], &[((2015, 1), 4), ((2015, 2), 7)]);
    assert_eq!(&wu_2015[wu_2015.len()-2..], &[((2015, 52), 7), ((2015, 53), 4)]);
}

fn test_by_week() {
    let mut weeks = dates_in_year(2013).__(by_week);
    assert_eq!(
        &*weeks.next().unwrap().1.collect::<Vec<_>>(),
        &[
            NaiveDate::from_ymd(2013, 1, 1),
            NaiveDate::from_ymd(2013, 1, 2),
            NaiveDate::from_ymd(2013, 1, 3),
            NaiveDate::from_ymd(2013, 1, 4),
            NaiveDate::from_ymd(2013, 1, 5),
        ]
    );
    assert_eq!(
        &*weeks.next().unwrap().1.collect::<Vec<_>>(),
        &[
            NaiveDate::from_ymd(2013, 1, 6),
            NaiveDate::from_ymd(2013, 1, 7),
            NaiveDate::from_ymd(2013, 1, 8),
            NaiveDate::from_ymd(2013, 1, 9),
            NaiveDate::from_ymd(2013, 1, 10),
            NaiveDate::from_ymd(2013, 1, 11),
            NaiveDate::from_ymd(2013, 1, 12),
        ]
    );
    assert_eq!(weeks.next().unwrap().1.nth(0).unwrap(), NaiveDate::from_ymd(2013, 1, 13));
}

/// The number of columns per day in the formatted output.
const COLS_PER_DAY: u32 = 3;

/// The number of columns per week in the formatted output.
const COLS_PER_WEEK: u32 = 7 * COLS_PER_DAY;

///
/// Formats an iterator of weeks into an iterator of strings.
///
fn format_weeks<It>(it: It) -> impl Iterator<Item=String>
where It: Iterator, It::Item: DateIterator {
    it.map(|week| {
        let mut buf = String::with_capacity((COLS_PER_DAY * COLS_PER_WEEK + 2) as usize);

        // Format each day into its own cell and append to target string.
        let mut last_day = 0;
        let mut first = true;
        for d in week {
            last_day = d.weekday().num_days_from_sunday();

            // Insert enough filler to align the first day with its respective day-of-week.
            if first {
                buf.extend(spaces((COLS_PER_DAY * last_day) as usize));
                first = false;
            }

            write!(buf, " {:>2}", d.day());
        }

        // Insert more filler at the end to fill up the remainder of the week,
        // if its a short week (e.g. at the end of the month).
        buf.extend(spaces((COLS_PER_DAY * (6 - last_day)) as usize));
        buf
    })
}

fn test_format_weeks() {
    let jan_2013 = dates_in_year(2013)
        .__(by_month).next() // pick January 2013 for testing purposes
        // NOTE: This `map` is because `next` returns an `Option<_>`.
        .map(|(_, month)|
            month.__(by_week)
                 .map(|(_, weeks)| weeks)
                 .__(format_weeks)
                 .join("\n"));

    assert_eq!(
        jan_2013.as_ref().map(|s| &**s),
        Some("        1  2  3  4  5\n\
           \x20 6  7  8  9 10 11 12\n\
           \x2013 14 15 16 17 18 19\n\
           \x2020 21 22 23 24 25 26\n\
           \x2027 28 29 30 31      ")
    );
}

///
/// Formats the name of a month, centered on COLS_PER_WEEK.
///
fn month_title(month: u32) -> String {
    const MONTH_NAMES: &'static [&'static str] = &[
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ];
    assert_eq!(MONTH_NAMES.len(), 12);

    // Determine how many spaces before and after the month name
    // we need to center it over the formatted weeks in the month.
    let name = MONTH_NAMES[(month - 1) as usize];
    assert!(name.len() < COLS_PER_WEEK as usize);
    let before = (COLS_PER_WEEK as usize - name.len()) / 2;
    let after = COLS_PER_WEEK as usize - name.len() - before;

    // NOTE: Being slightly more verbose to avoid extra allocations.
    let mut result = String::with_capacity(COLS_PER_WEEK as usize);
    result.extend(spaces(before));
    result.push_str(name);
    result.extend(spaces(after));
    result
}

fn test_month_title() {
    assert_eq!(month_title(1).len(), COLS_PER_WEEK as usize);
}

///
/// Formats a month.
///
fn format_month<It: DateIterator>(it: It) -> impl Iterator<Item=String> {
    let mut month_days = it.peekable();
    let title = month_title(month_days.peek().unwrap().month());

    Some(title).into_iter()
        .chain(month_days.__(by_week)
            .map(|(_, week)| week)
            .__(format_weeks))
}

fn test_format_month() {
    let month_fmt = dates_in_year(2013)
        .__(by_month).next() // Pick January as a test case
        .map(|(_, days)| days.into_iter()
            .__(format_month)
            .join("\n"));

    assert_eq!(
        month_fmt.as_ref().map(|s| &**s),
        Some("       January       \n\
           \x20       1  2  3  4  5\n\
           \x20 6  7  8  9 10 11 12\n\
           \x2013 14 15 16 17 18 19\n\
           \x2020 21 22 23 24 25 26\n\
           \x2027 28 29 30 31      ")
    );
}


///
/// Formats an iterator of months.
///
fn format_months<It>(it: It) -> impl Iterator<Item=impl Iterator<Item=String>>
where It: Iterator, It::Item: DateIterator {
    it.map(format_month)
}

///
/// Takes an iterator of iterators of strings; the sub-iterators are consumed
/// in lock-step, with their elements joined together.
///
trait PasteBlocks: Iterator + Sized
where Self::Item: Iterator<Item=String> {
    fn paste_blocks(self, sep_width: usize) -> PasteBlocksIter<Self::Item> {
        PasteBlocksIter {
            iters: self.collect(),
            cache: vec![],
            col_widths: None,
            sep_width: sep_width,
        }
    }
}

impl<It> PasteBlocks for It where It: Iterator, It::Item: Iterator<Item=String> {}

struct PasteBlocksIter<StrIt>
where StrIt: Iterator<Item=String> {
    iters: Vec<StrIt>,
    cache: Vec<Option<String>>,
    col_widths: Option<Vec<usize>>,
    sep_width: usize,
}

impl<StrIt> Iterator for PasteBlocksIter<StrIt>
where StrIt: Iterator<Item=String> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        self.cache.clear();

        // `cache` is now the next line from each iterator.
        self.cache.extend(self.iters.iter_mut().map(|it| it.next()));

        // If every line in `cache` is `None`, we have nothing further to do.
        if self.cache.iter().all(|e| e.is_none()) { return None }

        // Get the column widths if we haven't already.
        let col_widths = match self.col_widths {
            Some(ref v) => &**v,
            None => {
                self.col_widths = Some(self.cache.iter()
                    .map(|ms| ms.as_ref().map(|s| s.len()).unwrap_or(0))
                    .collect());
                &**self.col_widths.as_ref().unwrap()
            }
        };

        // Fill in any `None`s with spaces.
        let mut parts = col_widths.iter().cloned().zip(self.cache.iter_mut())
            .map(|(w,ms)| ms.take().unwrap_or_else(|| spaces(w).collect()));

        // Join them all together.
        let first = parts.next().unwrap_or(String::new());
        let sep_width = self.sep_width;
        Some(parts.fold(first, |mut accum, next| {
            accum.extend(spaces(sep_width));
            accum.push_str(&next);
            accum
        }))
    }
}

fn test_paste_blocks() {
    let row = dates_in_year(2013)
        .__(by_month).map(|(_, days)| days)
        .take(3)
        .__(format_months)
        .paste_blocks(1)
        .join("\n");
    assert_eq!(
        &*row,
        "       January              February                March        \n\
      \x20       1  2  3  4  5                  1  2                  1  2\n\
      \x20 6  7  8  9 10 11 12   3  4  5  6  7  8  9   3  4  5  6  7  8  9\n\
      \x2013 14 15 16 17 18 19  10 11 12 13 14 15 16  10 11 12 13 14 15 16\n\
      \x2020 21 22 23 24 25 26  17 18 19 20 21 22 23  17 18 19 20 21 22 23\n\
      \x2027 28 29 30 31        24 25 26 27 28        24 25 26 27 28 29 30\n\
      \x20                                            31                  "
    );
}

///
/// Produces an iterator that yields `n` elements at a time.
///
trait Chunks: Iterator + Sized {
    fn chunks(self, n: usize) -> ChunksIter<Self> {
        assert!(n > 0);
        ChunksIter {
            it: self,
            n: n,
        }
    }
}

impl<It> Chunks for It where It: Iterator {}

struct ChunksIter<It>
where It: Iterator {
    it: It,
    n: usize,
}

// NOTE: `chunks` in Rust is more-or-less impossible without overhead of some kind.
// Aliasing rules mean you need to add dynamic borrow checking, and the design of
// `Iterator` means that you need to have the iterator's state kept in an allocation
// that is jointly owned by the iterator itself and the sub-iterator.
// As such, I've chosen to cop-out and just heap-allocate each chunk.

impl<It> Iterator for ChunksIter<It>
where It: Iterator {
    type Item = Vec<It::Item>;

    fn next(&mut self) -> Option<Vec<It::Item>> {
        let first = match self.it.next() {
            Some(e) => e,
            None => return None
        };

        let mut result = Vec::with_capacity(self.n);
        result.push(first);

        Some((&mut self.it).take(self.n-1)
            .fold(result, |mut acc, next| { acc.push(next); acc }))
    }
}

fn test_chunks() {
    let r = &[1, 2, 3, 4, 5, 6, 7];
    let c = r.iter().cloned().chunks(3).collect::<Vec<_>>();
    assert_eq!(&*c, &[vec![1, 2, 3], vec![4, 5, 6], vec![7]]);
}

///
/// Formats a year.
///
fn format_year(year: i32, months_per_row: usize) -> String {
    const COL_SPACING: usize = 1;

    // Start by generating all dates for the given year.
    dates_in_year(year)

        // Group them by month and throw away month number.
        .__(by_month).map(|(_, days)| days)

        // Group the months into horizontal rows.
        .chunks(months_per_row)

        // Format each row
        .map(|r| r.into_iter()
            // By formatting each month
            .__(format_months)

            // Horizontally pasting each respective month's lines together.
            .paste_blocks(COL_SPACING)
            .join("\n")
        )

        // Insert a blank line between each row
        .join("\n\n")
}

fn test_format_year() {
    const MONTHS_PER_ROW: usize = 3;

    macro_rules! assert_eq_cal {
        ($lhs:expr, $rhs:expr) => {
            if $lhs != $rhs {
                println!("got:\n```\n{}\n```\n", $lhs.replace(" ", "."));
                println!("expected:\n```\n{}\n```", $rhs.replace(" ", "."));
                panic!("calendars didn't match!");
            }
        }
    }

    assert_eq_cal!(&format_year(1984, MONTHS_PER_ROW), "\
\x20      January              February                March        \n\
\x20 1  2  3  4  5  6  7            1  2  3  4               1  2  3\n\
\x20 8  9 10 11 12 13 14   5  6  7  8  9 10 11   4  5  6  7  8  9 10\n\
\x2015 16 17 18 19 20 21  12 13 14 15 16 17 18  11 12 13 14 15 16 17\n\
\x2022 23 24 25 26 27 28  19 20 21 22 23 24 25  18 19 20 21 22 23 24\n\
\x2029 30 31              26 27 28 29           25 26 27 28 29 30 31\n\
\n\
\x20       April                  May                  June         \n\
\x20 1  2  3  4  5  6  7         1  2  3  4  5                  1  2\n\
\x20 8  9 10 11 12 13 14   6  7  8  9 10 11 12   3  4  5  6  7  8  9\n\
\x2015 16 17 18 19 20 21  13 14 15 16 17 18 19  10 11 12 13 14 15 16\n\
\x2022 23 24 25 26 27 28  20 21 22 23 24 25 26  17 18 19 20 21 22 23\n\
\x2029 30                 27 28 29 30 31        24 25 26 27 28 29 30\n\
\n\
\x20       July                 August               September      \n\
\x20 1  2  3  4  5  6  7            1  2  3  4                     1\n\
\x20 8  9 10 11 12 13 14   5  6  7  8  9 10 11   2  3  4  5  6  7  8\n\
\x2015 16 17 18 19 20 21  12 13 14 15 16 17 18   9 10 11 12 13 14 15\n\
\x2022 23 24 25 26 27 28  19 20 21 22 23 24 25  16 17 18 19 20 21 22\n\
\x2029 30 31              26 27 28 29 30 31     23 24 25 26 27 28 29\n\
\x20                                            30                  \n\
\n\
\x20      October              November              December       \n\
\x20    1  2  3  4  5  6               1  2  3                     1\n\
\x20 7  8  9 10 11 12 13   4  5  6  7  8  9 10   2  3  4  5  6  7  8\n\
\x2014 15 16 17 18 19 20  11 12 13 14 15 16 17   9 10 11 12 13 14 15\n\
\x2021 22 23 24 25 26 27  18 19 20 21 22 23 24  16 17 18 19 20 21 22\n\
\x2028 29 30 31           25 26 27 28 29 30     23 24 25 26 27 28 29\n\
\x20                                            30 31               ");

    assert_eq_cal!(&format_year(2015, MONTHS_PER_ROW), "\
\x20      January              February                March        \n\
\x20             1  2  3   1  2  3  4  5  6  7   1  2  3  4  5  6  7\n\
\x20 4  5  6  7  8  9 10   8  9 10 11 12 13 14   8  9 10 11 12 13 14\n\
\x2011 12 13 14 15 16 17  15 16 17 18 19 20 21  15 16 17 18 19 20 21\n\
\x2018 19 20 21 22 23 24  22 23 24 25 26 27 28  22 23 24 25 26 27 28\n\
\x2025 26 27 28 29 30 31                        29 30 31            \n\
\n\
\x20       April                  May                  June         \n\
\x20          1  2  3  4                  1  2      1  2  3  4  5  6\n\
\x20 5  6  7  8  9 10 11   3  4  5  6  7  8  9   7  8  9 10 11 12 13\n\
\x2012 13 14 15 16 17 18  10 11 12 13 14 15 16  14 15 16 17 18 19 20\n\
\x2019 20 21 22 23 24 25  17 18 19 20 21 22 23  21 22 23 24 25 26 27\n\
\x2026 27 28 29 30        24 25 26 27 28 29 30  28 29 30            \n\
\x20                      31                                        \n\
\n\
\x20       July                 August               September      \n\
\x20          1  2  3  4                     1         1  2  3  4  5\n\
\x20 5  6  7  8  9 10 11   2  3  4  5  6  7  8   6  7  8  9 10 11 12\n\
\x2012 13 14 15 16 17 18   9 10 11 12 13 14 15  13 14 15 16 17 18 19\n\
\x2019 20 21 22 23 24 25  16 17 18 19 20 21 22  20 21 22 23 24 25 26\n\
\x2026 27 28 29 30 31     23 24 25 26 27 28 29  27 28 29 30         \n\
\x20                      30 31                                     \n\
\n\
\x20      October              November              December       \n\
\x20             1  2  3   1  2  3  4  5  6  7         1  2  3  4  5\n\
\x20 4  5  6  7  8  9 10   8  9 10 11 12 13 14   6  7  8  9 10 11 12\n\
\x2011 12 13 14 15 16 17  15 16 17 18 19 20 21  13 14 15 16 17 18 19\n\
\x2018 19 20 21 22 23 24  22 23 24 25 26 27 28  20 21 22 23 24 25 26\n\
\x2025 26 27 28 29 30 31  29 30                 27 28 29 30 31      ");
}

fn main() {
    // Run tests.
    test_spaces();
    test_dates_in_year();
    test_group_by();
    test_by_month();
    test_isoweekdate();
    test_by_week();
    test_format_weeks();
    test_month_title();
    test_format_month();
    test_paste_blocks();
    test_chunks();
    test_format_year();
}
