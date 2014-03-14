// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Simple getopt alternative.
//!
//! Construct a vector of options, either by using `reqopt`, `optopt`, and `optflag`
//! or by building them from components yourself, and pass them to `getopts`,
//! along with a vector of actual arguments (not including `argv[0]`). You'll
//! either get a failure code back, or a match. You'll have to verify whether
//! the amount of 'free' arguments in the match is what you expect. Use `opt_*`
//! accessors to get argument values out of the matches object.
//!
//! Single-character options are expected to appear on the command line with a
//! single preceding dash; multiple-character options are expected to be
//! proceeded by two dashes. Options that expect an argument accept their
//! argument following either a space or an equals sign. Single-character
//! options don't require the space.
//!
//! # Example
//!
//! The following example shows simple command line parsing for an application
//! that requires an input file to be specified, accepts an optional output
//! file name following `-o`, and accepts both `-h` and `--help` as optional flags.
//!
//! ~~~{.rust}
//! extern crate getopts;
//! use getopts::{optopt,optflag,getopts,OptGroup};
//! use std::os;
//!
//! fn do_work(inp: &str, out: Option<~str>) {
//!     println!("{}", inp);
//!     match out {
//!         Some(x) => println!("{}", x),
//!         None => println!("No Output"),
//!     }
//! }
//!
//! fn print_usage(program: &str, _opts: &[OptGroup]) {
//!     println!("Usage: {} [options]", program);
//!     println!("-o\t\tOutput");
//!     println!("-h --help\tUsage");
//! }
//!
//! fn main() {
//!     let args = os::args();
//!
//!     let program = args[0].clone();
//!
//!     let opts = ~[
//!         optopt("o", "", "set output file name", "NAME"),
//!         optflag("h", "help", "print this help menu")
//!     ];
//!     let matches = match getopts(args.tail(), opts) {
//!         Ok(m) => { m }
//!         Err(f) => { fail!(f.to_err_msg()) }
//!     };
//!     if matches.opt_present("h") {
//!         print_usage(program, opts);
//!         return;
//!     }
//!     let output = matches.opt_str("o");
//!     let input: &str = if !matches.free.is_empty() {
//!         matches.free[0].clone()
//!     } else {
//!         print_usage(program, opts);
//!         return;
//!     };
//!     do_work(input, output);
//! }
//! ~~~

#[crate_id = "getopts#0.10-pre"];
#[crate_type = "rlib"];
#[crate_type = "dylib"];
#[license = "MIT/ASL2"];
#[allow(missing_doc)];
#[allow(deprecated_owned_vector)];

#[feature(globs)];

use std::cmp::Eq;
use std::result::{Err, Ok};
use std::result;
use std::option::{Some, None};
use std::vec;

/// Name of an option. Either a string or a single char.
#[deriving(Clone, Eq)]
#[allow(missing_doc)]
pub enum Name {
    Long(~str),
    Short(char),
}

/// Describes whether an option has an argument.
#[deriving(Clone, Eq)]
#[allow(missing_doc)]
pub enum HasArg {
    Yes,
    No,
    Maybe,
}

/// Describes how often an option may occur.
#[deriving(Clone, Eq)]
#[allow(missing_doc)]
pub enum Occur {
    Req,
    Optional,
    Multi,
}

/// A description of a possible option.
#[deriving(Clone, Eq)]
pub struct Opt {
    /// Name of the option
    name: Name,
    /// Whether it has an argument
    hasarg: HasArg,
    /// How often it can occur
    occur: Occur,
    /// Which options it aliases
    priv aliases: ~[Opt],
}

/// One group of options, e.g., both -h and --help, along with
/// their shared description and properties.
#[deriving(Clone, Eq)]
pub struct OptGroup {
    /// Short Name of the `OptGroup`
    short_name: ~str,
    /// Long Name of the `OptGroup`
    long_name: ~str,
    /// Hint
    hint: ~str,
    /// Description
    desc: ~str,
    /// Whether it has an argument
    hasarg: HasArg,
    /// How often it can occur
    occur: Occur
}

/// Describes wether an option is given at all or has a value.
#[deriving(Clone, Eq)]
enum Optval {
    Val(~str),
    Given,
}

/// The result of checking command line arguments. Contains a vector
/// of matches and a vector of free strings.
#[deriving(Clone, Eq)]
pub struct Matches {
    /// Options that matched
    priv opts: ~[Opt],
    /// Values of the Options that matched
    priv vals: ~[~[Optval]],
    /// Free string fragments
    free: ~[~str]
}

/// The type returned when the command line does not conform to the
/// expected format. Call the `to_err_msg` method to retrieve the
/// error as a string.
#[deriving(Clone, Eq, Show)]
#[allow(missing_doc)]
pub enum Fail_ {
    ArgumentMissing(~str),
    UnrecognizedOption(~str),
    OptionMissing(~str),
    OptionDuplicated(~str),
    UnexpectedArgument(~str),
}

/// The type of failure that occurred.
#[deriving(Eq)]
#[allow(missing_doc)]
pub enum FailType {
    ArgumentMissing_,
    UnrecognizedOption_,
    OptionMissing_,
    OptionDuplicated_,
    UnexpectedArgument_,
}

/// The result of parsing a command line with a set of options.
pub type Result = result::Result<Matches, Fail_>;

impl Name {
    fn from_str(nm: &str) -> Name {
        if nm.len() == 1u {
            Short(nm.char_at(0u))
        } else {
            Long(nm.to_owned())
        }
    }

    fn to_str(&self) -> ~str {
        match *self {
            Short(ch) => ch.to_str(),
            Long(ref s) => s.to_owned()
        }
    }
}

impl OptGroup {
    /// Translate OptGroup into Opt.
    /// (Both short and long names correspond to different Opts).
    pub fn long_to_short(&self) -> Opt {
        let OptGroup {
            short_name: short_name,
            long_name: long_name,
            hasarg: hasarg,
            occur: occur,
            ..
        } = (*self).clone();

        match (short_name.len(), long_name.len()) {
            (0,0) => fail!("this long-format option was given no name"),
            (0,_) => Opt {
                name: Long((long_name)),
                hasarg: hasarg,
                occur: occur,
                aliases: ~[]
            },
            (1,0) => Opt {
                name: Short(short_name.char_at(0)),
                hasarg: hasarg,
                occur: occur,
                aliases: ~[]
            },
            (1,_) => Opt {
                name: Long((long_name)),
                hasarg: hasarg,
                occur:  occur,
                aliases: ~[
                    Opt {
                        name: Short(short_name.char_at(0)),
                        hasarg: hasarg,
                        occur:  occur,
                        aliases: ~[]
                    }
                ]
            },
            (_,_) => fail!("something is wrong with the long-form opt")
        }
    }
}

impl Matches {
    fn opt_vals(&self, nm: &str) -> ~[Optval] {
        match find_opt(self.opts, Name::from_str(nm)) {
            Some(id) => self.vals[id].clone(),
            None => fail!("No option '{}' defined", nm)
        }
    }

    fn opt_val(&self, nm: &str) -> Option<Optval> {
        let vals = self.opt_vals(nm);
        if vals.is_empty() {
            None
        } else {
            Some(vals[0].clone())
        }
    }

    /// Returns true if an option was matched.
    pub fn opt_present(&self, nm: &str) -> bool {
        !self.opt_vals(nm).is_empty()
    }

    /// Returns the number of times an option was matched.
    pub fn opt_count(&self, nm: &str) -> uint {
        self.opt_vals(nm).len()
    }

    /// Returns true if any of several options were matched.
    pub fn opts_present(&self, names: &[~str]) -> bool {
        for nm in names.iter() {
            match find_opt(self.opts, Name::from_str(*nm)) {
                Some(id) if !self.vals[id].is_empty() => return true,
                _ => (),
            };
        }
        false
    }

    /// Returns the string argument supplied to one of several matching options or `None`.
    pub fn opts_str(&self, names: &[~str]) -> Option<~str> {
        for nm in names.iter() {
            match self.opt_val(*nm) {
                Some(Val(ref s)) => return Some(s.clone()),
                _ => ()
            }
        }
        None
    }

    /// Returns a vector of the arguments provided to all matches of the given
    /// option.
    ///
    /// Used when an option accepts multiple values.
    pub fn opt_strs(&self, nm: &str) -> ~[~str] {
        let mut acc: ~[~str] = ~[];
        let r = self.opt_vals(nm);
        for v in r.iter() {
            match *v {
                Val(ref s) => acc.push((*s).clone()),
                _ => ()
            }
        }
        acc
    }

    /// Returns the string argument supplied to a matching option or `None`.
    pub fn opt_str(&self, nm: &str) -> Option<~str> {
        let vals = self.opt_vals(nm);
        if vals.is_empty() {
            return None::<~str>;
        }
        match vals[0] {
            Val(ref s) => Some((*s).clone()),
            _ => None
        }
    }


    /// Returns the matching string, a default, or none.
    ///
    /// Returns none if the option was not present, `def` if the option was
    /// present but no argument was provided, and the argument if the option was
    /// present and an argument was provided.
    pub fn opt_default(&self, nm: &str, def: &str) -> Option<~str> {
        let vals = self.opt_vals(nm);
        if vals.is_empty() { return None; }
        match vals[0] {
            Val(ref s) => Some((*s).clone()),
            _ => Some(def.to_owned())
        }
    }

}

fn is_arg(arg: &str) -> bool {
    arg.len() > 1 && arg[0] == '-' as u8
}

fn find_opt(opts: &[Opt], nm: Name) -> Option<uint> {
    // Search main options.
    let pos = opts.iter().position(|opt| opt.name == nm);
    if pos.is_some() {
        return pos
    }

    // Search in aliases.
    for candidate in opts.iter() {
        if candidate.aliases.iter().position(|opt| opt.name == nm).is_some() {
            return opts.iter().position(|opt| opt.name == candidate.name);
        }
    }

    None
}

/// Create a long option that is required and takes an argument.
pub fn reqopt(short_name: &str, long_name: &str, desc: &str, hint: &str) -> OptGroup {
    let len = short_name.len();
    assert!(len == 1 || len == 0);
    OptGroup {
        short_name: short_name.to_owned(),
        long_name: long_name.to_owned(),
        hint: hint.to_owned(),
        desc: desc.to_owned(),
        hasarg: Yes,
        occur: Req
    }
}

/// Create a long option that is optional and takes an argument.
pub fn optopt(short_name: &str, long_name: &str, desc: &str, hint: &str) -> OptGroup {
    let len = short_name.len();
    assert!(len == 1 || len == 0);
    OptGroup {
        short_name: short_name.to_owned(),
        long_name: long_name.to_owned(),
        hint: hint.to_owned(),
        desc: desc.to_owned(),
        hasarg: Yes,
        occur: Optional
    }
}

/// Create a long option that is optional and does not take an argument.
pub fn optflag(short_name: &str, long_name: &str, desc: &str) -> OptGroup {
    let len = short_name.len();
    assert!(len == 1 || len == 0);
    OptGroup {
        short_name: short_name.to_owned(),
        long_name: long_name.to_owned(),
        hint: ~"",
        desc: desc.to_owned(),
        hasarg: No,
        occur: Optional
    }
}

/// Create a long option that can occur more than once and does not
/// take an argument.
pub fn optflagmulti(short_name: &str, long_name: &str, desc: &str) -> OptGroup {
    let len = short_name.len();
    assert!(len == 1 || len == 0);
    OptGroup {
        short_name: short_name.to_owned(),
        long_name: long_name.to_owned(),
        hint: ~"",
        desc: desc.to_owned(),
        hasarg: No,
        occur: Multi
    }
}

/// Create a long option that is optional and takes an optional argument.
pub fn optflagopt(short_name: &str, long_name: &str, desc: &str, hint: &str) -> OptGroup {
    let len = short_name.len();
    assert!(len == 1 || len == 0);
    OptGroup {
        short_name: short_name.to_owned(),
        long_name: long_name.to_owned(),
        hint: hint.to_owned(),
        desc: desc.to_owned(),
        hasarg: Maybe,
        occur: Optional
    }
}

/// Create a long option that is optional, takes an argument, and may occur
/// multiple times.
pub fn optmulti(short_name: &str, long_name: &str, desc: &str, hint: &str) -> OptGroup {
    let len = short_name.len();
    assert!(len == 1 || len == 0);
    OptGroup {
        short_name: short_name.to_owned(),
        long_name: long_name.to_owned(),
        hint: hint.to_owned(),
        desc: desc.to_owned(),
        hasarg: Yes,
        occur: Multi
    }
}

/// Create a generic option group, stating all parameters explicitly
pub fn opt(short_name: &str,
           long_name: &str,
           desc: &str,
           hint: &str,
           hasarg: HasArg,
           occur: Occur) -> OptGroup {
    let len = short_name.len();
    assert!(len == 1 || len == 0);
    OptGroup {
        short_name: short_name.to_owned(),
        long_name: long_name.to_owned(),
        hint: hint.to_owned(),
        desc: desc.to_owned(),
        hasarg: hasarg,
        occur: occur
    }
}

impl Fail_ {
    /// Convert a `Fail_` enum into an error string.
    pub fn to_err_msg(self) -> ~str {
        match self {
            ArgumentMissing(ref nm) => {
                format!("Argument to option '{}' missing.", *nm)
            }
            UnrecognizedOption(ref nm) => {
                format!("Unrecognized option: '{}'.", *nm)
            }
            OptionMissing(ref nm) => {
                format!("Required option '{}' missing.", *nm)
            }
            OptionDuplicated(ref nm) => {
                format!("Option '{}' given more than once.", *nm)
            }
            UnexpectedArgument(ref nm) => {
                format!("Option '{}' does not take an argument.", *nm)
            }
        }
    }
}

/// Parse command line arguments according to the provided options.
///
/// On success returns `Ok(Opt)`. Use methods such as `opt_present`
/// `opt_str`, etc. to interrogate results.  Returns `Err(Fail_)` on failure.
/// Use `to_err_msg` to get an error message.
pub fn getopts(args: &[~str], optgrps: &[OptGroup]) -> Result {
    let opts = optgrps.map(|x| x.long_to_short());
    let n_opts = opts.len();

    fn f(_x: uint) -> ~[Optval] { return ~[]; }

    let mut vals = vec::from_fn(n_opts, f);
    let mut free: ~[~str] = ~[];
    let l = args.len();
    let mut i = 0;
    while i < l {
        let cur = args[i].clone();
        let curlen = cur.len();
        if !is_arg(cur) {
            free.push(cur);
        } else if cur == ~"--" {
            let mut j = i + 1;
            while j < l { free.push(args[j].clone()); j += 1; }
            break;
        } else {
            let mut names;
            let mut i_arg = None;
            if cur[1] == '-' as u8 {
                let tail = cur.slice(2, curlen);
                let tail_eq: ~[&str] = tail.split('=').collect();
                if tail_eq.len() <= 1 {
                    names = ~[Long(tail.to_owned())];
                } else {
                    names =
                        ~[Long(tail_eq[0].to_owned())];
                    i_arg = Some(tail_eq[1].to_owned());
                }
            } else {
                let mut j = 1;
                let mut last_valid_opt_id = None;
                names = ~[];
                while j < curlen {
                    let range = cur.char_range_at(j);
                    let opt = Short(range.ch);

                    /* In a series of potential options (eg. -aheJ), if we
                       see one which takes an argument, we assume all
                       subsequent characters make up the argument. This
                       allows options such as -L/usr/local/lib/foo to be
                       interpreted correctly
                    */

                    match find_opt(opts, opt.clone()) {
                      Some(id) => last_valid_opt_id = Some(id),
                      None => {
                        let arg_follows =
                            last_valid_opt_id.is_some() &&
                            match opts[last_valid_opt_id.unwrap()]
                              .hasarg {

                              Yes | Maybe => true,
                              No => false
                            };
                        if arg_follows && j < curlen {
                            i_arg = Some(cur.slice(j, curlen).to_owned());
                            break;
                        } else {
                            last_valid_opt_id = None;
                        }
                      }
                    }
                    names.push(opt);
                    j = range.next;
                }
            }
            let mut name_pos = 0;
            for nm in names.iter() {
                name_pos += 1;
                let optid = match find_opt(opts, (*nm).clone()) {
                  Some(id) => id,
                  None => return Err(UnrecognizedOption(nm.to_str()))
                };
                match opts[optid].hasarg {
                  No => {
                    if !i_arg.is_none() {
                        return Err(UnexpectedArgument(nm.to_str()));
                    }
                    vals[optid].push(Given);
                  }
                  Maybe => {
                    if !i_arg.is_none() {
                        vals[optid].push(Val((i_arg.clone()).unwrap()));
                    } else if name_pos < names.len() ||
                                  i + 1 == l || is_arg(args[i + 1]) {
                        vals[optid].push(Given);
                    } else { i += 1; vals[optid].push(Val(args[i].clone())); }
                  }
                  Yes => {
                    if !i_arg.is_none() {
                        vals[optid].push(Val(i_arg.clone().unwrap()));
                    } else if i + 1 == l {
                        return Err(ArgumentMissing(nm.to_str()));
                    } else { i += 1; vals[optid].push(Val(args[i].clone())); }
                  }
                }
            }
        }
        i += 1;
    }
    i = 0u;
    while i < n_opts {
        let n = vals[i].len();
        let occ = opts[i].occur;
        if occ == Req {
            if n == 0 {
                return Err(OptionMissing(opts[i].name.to_str()));
            }
        }
        if occ != Multi {
            if n > 1 {
                return Err(OptionDuplicated(opts[i].name.to_str()));
            }
        }
        i += 1;
    }
    Ok(Matches {
        opts: opts.to_owned(),
        vals: vals,
        free: free
    })
}

/// Derive a usage message from a set of long options.
pub fn usage(brief: &str, opts: &[OptGroup]) -> ~str {

    let desc_sep = "\n" + " ".repeat(24);

    let mut rows = opts.iter().map(|optref| {
        let OptGroup{short_name: short_name,
                     long_name: long_name,
                     hint: hint,
                     desc: desc,
                     hasarg: hasarg,
                     ..} = (*optref).clone();

        let mut row = " ".repeat(4);

        // short option
        match short_name.len() {
            0 => {}
            1 => {
                row.push_char('-');
                row.push_str(short_name);
                row.push_char(' ');
            }
            _ => fail!("the short name should only be 1 ascii char long"),
        }

        // long option
        match long_name.len() {
            0 => {}
            _ => {
                row.push_str("--");
                row.push_str(long_name);
                row.push_char(' ');
            }
        }

        // arg
        match hasarg {
            No => {}
            Yes => row.push_str(hint),
            Maybe => {
                row.push_char('[');
                row.push_str(hint);
                row.push_char(']');
            }
        }

        // FIXME: #5516 should be graphemes not codepoints
        // here we just need to indent the start of the description
        let rowlen = row.char_len();
        if rowlen < 24 {
            for _ in range(0, 24 - rowlen) {
                row.push_char(' ');
            }
        } else {
            row.push_str(desc_sep)
        }

        // Normalize desc to contain words separated by one space character
        let mut desc_normalized_whitespace = ~"";
        for word in desc.words() {
            desc_normalized_whitespace.push_str(word);
            desc_normalized_whitespace.push_char(' ');
        }

        // FIXME: #5516 should be graphemes not codepoints
        let mut desc_rows = ~[];
        each_split_within(desc_normalized_whitespace, 54, |substr| {
            desc_rows.push(substr.to_owned());
            true
        });

        // FIXME: #5516 should be graphemes not codepoints
        // wrapped description
        row.push_str(desc_rows.connect(desc_sep));

        row
    });

    format!("{}\n\nOptions:\n{}\n", brief, rows.collect::<~[~str]>().connect("\n"))
}

fn format_option(opt: &OptGroup) -> ~str {
    let mut line = ~"";

    if opt.occur != Req {
        line.push_char('[');
    }

    // Use short_name is possible, but fallback to long_name.
    if opt.short_name.len() > 0 {
        line.push_char('-');
        line.push_str(opt.short_name);
    } else {
        line.push_str("--");
        line.push_str(opt.long_name);
    }

    if opt.hasarg != No {
        line.push_char(' ');
        if opt.hasarg == Maybe {
            line.push_char('[');
        }
        line.push_str(opt.hint);
        if opt.hasarg == Maybe {
            line.push_char(']');
        }
    }

    if opt.occur != Req {
        line.push_char(']');
    }
    if opt.occur == Multi {
        line.push_str("..");
    }

    line
}

/// Derive a short one-line usage summary from a set of long options.
pub fn short_usage(program_name: &str, opts: &[OptGroup]) -> ~str {
    let mut line = ~"Usage: " + program_name + " ";
    line.push_str(opts.iter().map(format_option).to_owned_vec().connect(" "));

    line
}


/// Splits a string into substrings with possibly internal whitespace,
/// each of them at most `lim` bytes long. The substrings have leading and trailing
/// whitespace removed, and are only cut at whitespace boundaries.
///
/// Note: Function was moved here from `std::str` because this module is the only place that
/// uses it, and because it was to specific for a general string function.
///
/// #Failure:
///
/// Fails during iteration if the string contains a non-whitespace
/// sequence longer than the limit.
fn each_split_within<'a>(ss: &'a str, lim: uint, it: |&'a str| -> bool)
                     -> bool {
    // Just for fun, let's write this as a state machine:

    enum SplitWithinState {
        A,  // leading whitespace, initial state
        B,  // words
        C,  // internal and trailing whitespace
    }
    enum Whitespace {
        Ws, // current char is whitespace
        Cr  // current char is not whitespace
    }
    enum LengthLimit {
        UnderLim, // current char makes current substring still fit in limit
        OverLim   // current char makes current substring no longer fit in limit
    }

    let mut slice_start = 0;
    let mut last_start = 0;
    let mut last_end = 0;
    let mut state = A;
    let mut fake_i = ss.len();
    let mut lim = lim;

    let mut cont = true;

    // if the limit is larger than the string, lower it to save cycles
    if lim >= fake_i {
        lim = fake_i;
    }

    let machine: |&mut bool, (uint, char)| -> bool = |cont, (i, c)| {
        let whitespace = if ::std::char::is_whitespace(c) { Ws }       else { Cr };
        let limit      = if (i - slice_start + 1) <= lim  { UnderLim } else { OverLim };

        state = match (state, whitespace, limit) {
            (A, Ws, _)        => { A }
            (A, Cr, _)        => { slice_start = i; last_start = i; B }

            (B, Cr, UnderLim) => { B }
            (B, Cr, OverLim)  if (i - last_start + 1) > lim
                            => fail!("word starting with {} longer than limit!",
                                    ss.slice(last_start, i + 1)),
            (B, Cr, OverLim)  => {
                *cont = it(ss.slice(slice_start, last_end));
                slice_start = last_start;
                B
            }
            (B, Ws, UnderLim) => {
                last_end = i;
                C
            }
            (B, Ws, OverLim)  => {
                last_end = i;
                *cont = it(ss.slice(slice_start, last_end));
                A
            }

            (C, Cr, UnderLim) => {
                last_start = i;
                B
            }
            (C, Cr, OverLim)  => {
                *cont = it(ss.slice(slice_start, last_end));
                slice_start = i;
                last_start = i;
                last_end = i;
                B
            }
            (C, Ws, OverLim)  => {
                *cont = it(ss.slice(slice_start, last_end));
                A
            }
            (C, Ws, UnderLim) => {
                C
            }
        };

        *cont
    };

    ss.char_indices().advance(|x| machine(&mut cont, x));

    // Let the automaton 'run out' by supplying trailing whitespace
    while cont && match state { B | C => true, A => false } {
        machine(&mut cont, (fake_i, ' '));
        fake_i += 1;
    }
    return cont;
}

#[test]
fn test_split_within() {
    fn t(s: &str, i: uint, u: &[~str]) {
        let mut v = ~[];
        each_split_within(s, i, |s| { v.push(s.to_owned()); true });
        assert!(v.iter().zip(u.iter()).all(|(a,b)| a == b));
    }
    t("", 0, []);
    t("", 15, []);
    t("hello", 15, [~"hello"]);
    t("\nMary had a little lamb\nLittle lamb\n", 15,
        [~"Mary had a", ~"little lamb", ~"Little lamb"]);
    t("\nMary had a little lamb\nLittle lamb\n", ::std::uint::MAX,
        [~"Mary had a little lamb\nLittle lamb"]);
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::result::{Err, Ok};
    use std::result;

    fn check_fail_type(f: Fail_, ft: FailType) {
        match f {
          ArgumentMissing(_) => assert!(ft == ArgumentMissing_),
          UnrecognizedOption(_) => assert!(ft == UnrecognizedOption_),
          OptionMissing(_) => assert!(ft == OptionMissing_),
          OptionDuplicated(_) => assert!(ft == OptionDuplicated_),
          UnexpectedArgument(_) => assert!(ft == UnexpectedArgument_)
        }
    }

    // Tests for reqopt
    #[test]
    fn test_reqopt() {
        let long_args = ~[~"--test=20"];
        let opts = ~[reqopt("t", "test", "testing", "TEST")];
        let rs = getopts(long_args, opts);
        match rs {
          Ok(ref m) => {
            assert!(m.opt_present("test"));
            assert_eq!(m.opt_str("test").unwrap(), ~"20");
            assert!(m.opt_present("t"));
            assert_eq!(m.opt_str("t").unwrap(), ~"20");
          }
          _ => { fail!("test_reqopt failed (long arg)"); }
        }
        let short_args = ~[~"-t", ~"20"];
        match getopts(short_args, opts) {
          Ok(ref m) => {
            assert!((m.opt_present("test")));
            assert_eq!(m.opt_str("test").unwrap(), ~"20");
            assert!((m.opt_present("t")));
            assert_eq!(m.opt_str("t").unwrap(), ~"20");
          }
          _ => { fail!("test_reqopt failed (short arg)"); }
        }
    }

    #[test]
    fn test_reqopt_missing() {
        let args = ~[~"blah"];
        let opts = ~[reqopt("t", "test", "testing", "TEST")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionMissing_),
          _ => fail!()
        }
    }

    #[test]
    fn test_reqopt_no_arg() {
        let long_args = ~[~"--test"];
        let opts = ~[reqopt("t", "test", "testing", "TEST")];
        let rs = getopts(long_args, opts);
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
        let short_args = ~[~"-t"];
        match getopts(short_args, opts) {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
    }

    #[test]
    fn test_reqopt_multi() {
        let args = ~[~"--test=20", ~"-t", ~"30"];
        let opts = ~[reqopt("t", "test", "testing", "TEST")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail!()
        }
    }

    // Tests for optopt
    #[test]
    fn test_optopt() {
        let long_args = ~[~"--test=20"];
        let opts = ~[optopt("t", "test", "testing", "TEST")];
        let rs = getopts(long_args, opts);
        match rs {
          Ok(ref m) => {
            assert!(m.opt_present("test"));
            assert_eq!(m.opt_str("test").unwrap(), ~"20");
            assert!((m.opt_present("t")));
            assert_eq!(m.opt_str("t").unwrap(), ~"20");
          }
          _ => fail!()
        }
        let short_args = ~[~"-t", ~"20"];
        match getopts(short_args, opts) {
          Ok(ref m) => {
            assert!((m.opt_present("test")));
            assert_eq!(m.opt_str("test").unwrap(), ~"20");
            assert!((m.opt_present("t")));
            assert_eq!(m.opt_str("t").unwrap(), ~"20");
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optopt_missing() {
        let args = ~[~"blah"];
        let opts = ~[optopt("t", "test", "testing", "TEST")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert!(!m.opt_present("test"));
            assert!(!m.opt_present("t"));
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optopt_no_arg() {
        let long_args = ~[~"--test"];
        let opts = ~[optopt("t", "test", "testing", "TEST")];
        let rs = getopts(long_args, opts);
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
        let short_args = ~[~"-t"];
        match getopts(short_args, opts) {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
    }

    #[test]
    fn test_optopt_multi() {
        let args = ~[~"--test=20", ~"-t", ~"30"];
        let opts = ~[optopt("t", "test", "testing", "TEST")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail!()
        }
    }

    // Tests for optflag
    #[test]
    fn test_optflag() {
        let long_args = ~[~"--test"];
        let opts = ~[optflag("t", "test", "testing")];
        let rs = getopts(long_args, opts);
        match rs {
          Ok(ref m) => {
            assert!(m.opt_present("test"));
            assert!(m.opt_present("t"));
          }
          _ => fail!()
        }
        let short_args = ~[~"-t"];
        match getopts(short_args, opts) {
          Ok(ref m) => {
            assert!(m.opt_present("test"));
            assert!(m.opt_present("t"));
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflag_missing() {
        let args = ~[~"blah"];
        let opts = ~[optflag("t", "test", "testing")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert!(!m.opt_present("test"));
            assert!(!m.opt_present("t"));
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflag_long_arg() {
        let args = ~[~"--test=20"];
        let opts = ~[optflag("t", "test", "testing")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => {
            error!("{:?}", f.clone().to_err_msg());
            check_fail_type(f, UnexpectedArgument_);
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflag_multi() {
        let args = ~[~"--test", ~"-t"];
        let opts = ~[optflag("t", "test", "testing")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail!()
        }
    }

    #[test]
    fn test_optflag_short_arg() {
        let args = ~[~"-t", ~"20"];
        let opts = ~[optflag("t", "test", "testing")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            // The next variable after the flag is just a free argument

            assert!(m.free[0] == ~"20");
          }
          _ => fail!()
        }
    }

    // Tests for optflagmulti
    #[test]
    fn test_optflagmulti_short1() {
        let args = ~[~"-v"];
        let opts = ~[optflagmulti("v", "verbose", "verbosity")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert_eq!(m.opt_count("v"), 1);
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflagmulti_short2a() {
        let args = ~[~"-v", ~"-v"];
        let opts = ~[optflagmulti("v", "verbose", "verbosity")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert_eq!(m.opt_count("v"), 2);
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflagmulti_short2b() {
        let args = ~[~"-vv"];
        let opts = ~[optflagmulti("v", "verbose", "verbosity")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert_eq!(m.opt_count("v"), 2);
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflagmulti_long1() {
        let args = ~[~"--verbose"];
        let opts = ~[optflagmulti("v", "verbose", "verbosity")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert_eq!(m.opt_count("verbose"), 1);
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflagmulti_long2() {
        let args = ~[~"--verbose", ~"--verbose"];
        let opts = ~[optflagmulti("v", "verbose", "verbosity")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert_eq!(m.opt_count("verbose"), 2);
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflagmulti_mix() {
        let args = ~[~"--verbose", ~"-v", ~"-vv", ~"verbose"];
        let opts = ~[optflagmulti("v", "verbose", "verbosity")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert_eq!(m.opt_count("verbose"), 4);
            assert_eq!(m.opt_count("v"), 4);
          }
          _ => fail!()
        }
    }

    // Tests for optmulti
    #[test]
    fn test_optmulti() {
        let long_args = ~[~"--test=20"];
        let opts = ~[optmulti("t", "test", "testing", "TEST")];
        let rs = getopts(long_args, opts);
        match rs {
          Ok(ref m) => {
            assert!((m.opt_present("test")));
            assert_eq!(m.opt_str("test").unwrap(), ~"20");
            assert!((m.opt_present("t")));
            assert_eq!(m.opt_str("t").unwrap(), ~"20");
          }
          _ => fail!()
        }
        let short_args = ~[~"-t", ~"20"];
        match getopts(short_args, opts) {
          Ok(ref m) => {
            assert!((m.opt_present("test")));
            assert_eq!(m.opt_str("test").unwrap(), ~"20");
            assert!((m.opt_present("t")));
            assert_eq!(m.opt_str("t").unwrap(), ~"20");
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optmulti_missing() {
        let args = ~[~"blah"];
        let opts = ~[optmulti("t", "test", "testing", "TEST")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert!(!m.opt_present("test"));
            assert!(!m.opt_present("t"));
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optmulti_no_arg() {
        let long_args = ~[~"--test"];
        let opts = ~[optmulti("t", "test", "testing", "TEST")];
        let rs = getopts(long_args, opts);
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
        let short_args = ~[~"-t"];
        match getopts(short_args, opts) {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
    }

    #[test]
    fn test_optmulti_multi() {
        let args = ~[~"--test=20", ~"-t", ~"30"];
        let opts = ~[optmulti("t", "test", "testing", "TEST")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
              assert!(m.opt_present("test"));
              assert_eq!(m.opt_str("test").unwrap(), ~"20");
              assert!(m.opt_present("t"));
              assert_eq!(m.opt_str("t").unwrap(), ~"20");
              let pair = m.opt_strs("test");
              assert!(pair[0] == ~"20");
              assert!(pair[1] == ~"30");
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_unrecognized_option() {
        let long_args = ~[~"--untest"];
        let opts = ~[optmulti("t", "test", "testing", "TEST")];
        let rs = getopts(long_args, opts);
        match rs {
          Err(f) => check_fail_type(f, UnrecognizedOption_),
          _ => fail!()
        }
        let short_args = ~[~"-u"];
        match getopts(short_args, opts) {
          Err(f) => check_fail_type(f, UnrecognizedOption_),
          _ => fail!()
        }
    }

    #[test]
    fn test_combined() {
        let args =
            ~[~"prog", ~"free1", ~"-s", ~"20", ~"free2",
              ~"--flag", ~"--long=30", ~"-f", ~"-m", ~"40",
              ~"-m", ~"50", ~"-n", ~"-A B", ~"-n", ~"-60 70"];
        let opts =
            ~[optopt("s", "something", "something", "SOMETHING"),
              optflag("", "flag", "a flag"),
              reqopt("", "long", "hi", "LONG"),
              optflag("f", "", "another flag"),
              optmulti("m", "", "mmmmmm", "YUM"),
              optmulti("n", "", "nothing", "NOTHING"),
              optopt("", "notpresent", "nothing to see here", "NOPE")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert!(m.free[0] == ~"prog");
            assert!(m.free[1] == ~"free1");
            assert_eq!(m.opt_str("s").unwrap(), ~"20");
            assert!(m.free[2] == ~"free2");
            assert!((m.opt_present("flag")));
            assert_eq!(m.opt_str("long").unwrap(), ~"30");
            assert!((m.opt_present("f")));
            let pair = m.opt_strs("m");
            assert!(pair[0] == ~"40");
            assert!(pair[1] == ~"50");
            let pair = m.opt_strs("n");
            assert!(pair[0] == ~"-A B");
            assert!(pair[1] == ~"-60 70");
            assert!((!m.opt_present("notpresent")));
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_multi() {
        let opts = ~[optopt("e", "", "encrypt", "ENCRYPT"),
                     optopt("", "encrypt", "encrypt", "ENCRYPT"),
                     optopt("f", "", "flag", "FLAG")];

        let args_single = ~[~"-e", ~"foo"];
        let matches_single = &match getopts(args_single, opts) {
          result::Ok(m) => m,
          result::Err(_) => fail!()
        };
        assert!(matches_single.opts_present([~"e"]));
        assert!(matches_single.opts_present([~"encrypt", ~"e"]));
        assert!(matches_single.opts_present([~"e", ~"encrypt"]));
        assert!(!matches_single.opts_present([~"encrypt"]));
        assert!(!matches_single.opts_present([~"thing"]));
        assert!(!matches_single.opts_present([]));

        assert_eq!(matches_single.opts_str([~"e"]).unwrap(), ~"foo");
        assert_eq!(matches_single.opts_str([~"e", ~"encrypt"]).unwrap(), ~"foo");
        assert_eq!(matches_single.opts_str([~"encrypt", ~"e"]).unwrap(), ~"foo");

        let args_both = ~[~"-e", ~"foo", ~"--encrypt", ~"foo"];
        let matches_both = &match getopts(args_both, opts) {
          result::Ok(m) => m,
          result::Err(_) => fail!()
        };
        assert!(matches_both.opts_present([~"e"]));
        assert!(matches_both.opts_present([~"encrypt"]));
        assert!(matches_both.opts_present([~"encrypt", ~"e"]));
        assert!(matches_both.opts_present([~"e", ~"encrypt"]));
        assert!(!matches_both.opts_present([~"f"]));
        assert!(!matches_both.opts_present([~"thing"]));
        assert!(!matches_both.opts_present([]));

        assert_eq!(matches_both.opts_str([~"e"]).unwrap(), ~"foo");
        assert_eq!(matches_both.opts_str([~"encrypt"]).unwrap(), ~"foo");
        assert_eq!(matches_both.opts_str([~"e", ~"encrypt"]).unwrap(), ~"foo");
        assert_eq!(matches_both.opts_str([~"encrypt", ~"e"]).unwrap(), ~"foo");
    }

    #[test]
    fn test_nospace() {
        let args = ~[~"-Lfoo", ~"-M."];
        let opts = ~[optmulti("L", "", "library directory", "LIB"),
                     optmulti("M", "", "something", "MMMM")];
        let matches = &match getopts(args, opts) {
          result::Ok(m) => m,
          result::Err(_) => fail!()
        };
        assert!(matches.opts_present([~"L"]));
        assert_eq!(matches.opts_str([~"L"]).unwrap(), ~"foo");
        assert!(matches.opts_present([~"M"]));
        assert_eq!(matches.opts_str([~"M"]).unwrap(), ~".");

    }

    #[test]
    fn test_long_to_short() {
        let mut short = Opt { name: Long(~"banana"),
                              hasarg: Yes,
                              occur: Req,
                              aliases: ~[] };
        short.aliases = ~[Opt { name: Short('b'),
                                hasarg: Yes,
                                occur: Req,
                                aliases: ~[] }];
        let verbose = reqopt("b", "banana", "some bananas", "VAL");

        assert!(verbose.long_to_short() == short);
    }

    #[test]
    fn test_aliases_long_and_short() {
        let opts = ~[
            optflagmulti("a", "apple", "Desc"),
        ];

        let args = ~[~"-a", ~"--apple", ~"-a"];

        let matches = getopts(args, opts).unwrap();
        assert_eq!(3, matches.opt_count("a"));
        assert_eq!(3, matches.opt_count("apple"));
    }

    #[test]
    fn test_usage() {
        let optgroups = ~[
            reqopt("b", "banana", "Desc", "VAL"),
            optopt("a", "012345678901234567890123456789",
                             "Desc", "VAL"),
            optflag("k", "kiwi", "Desc"),
            optflagopt("p", "", "Desc", "VAL"),
            optmulti("l", "", "Desc", "VAL"),
        ];

        let expected =
~"Usage: fruits

Options:
    -b --banana VAL     Desc
    -a --012345678901234567890123456789 VAL
                        Desc
    -k --kiwi           Desc
    -p [VAL]            Desc
    -l VAL              Desc
";

        let generated_usage = usage("Usage: fruits", optgroups);

        debug!("expected: <<{}>>", expected);
        debug!("generated: <<{}>>", generated_usage);
        assert_eq!(generated_usage, expected);
    }

    #[test]
    fn test_usage_description_wrapping() {
        // indentation should be 24 spaces
        // lines wrap after 78: or rather descriptions wrap after 54

        let optgroups = ~[
            optflag("k", "kiwi",
                "This is a long description which won't be wrapped..+.."), // 54
            optflag("a", "apple",
                "This is a long description which _will_ be wrapped..+.."), // 55
        ];

        let expected =
~"Usage: fruits

Options:
    -k --kiwi           This is a long description which won't be wrapped..+..
    -a --apple          This is a long description which _will_ be
                        wrapped..+..
";

        let usage = usage("Usage: fruits", optgroups);

        debug!("expected: <<{}>>", expected);
        debug!("generated: <<{}>>", usage);
        assert!(usage == expected)
    }

    #[test]
    fn test_usage_description_multibyte_handling() {
        let optgroups = ~[
            optflag("k", "k\u2013w\u2013",
                "The word kiwi is normally spelled with two i's"),
            optflag("a", "apple",
                "This \u201Cdescription\u201D has some characters that could \
confuse the line wrapping; an apple costs 0.51€ in some parts of Europe."),
        ];

        let expected =
~"Usage: fruits

Options:
    -k --k–w–           The word kiwi is normally spelled with two i's
    -a --apple          This “description” has some characters that could
                        confuse the line wrapping; an apple costs 0.51€ in
                        some parts of Europe.
";

        let usage = usage("Usage: fruits", optgroups);

        debug!("expected: <<{}>>", expected);
        debug!("generated: <<{}>>", usage);
        assert!(usage == expected)
    }

    #[test]
    fn test_short_usage() {
        let optgroups = ~[
            reqopt("b", "banana", "Desc", "VAL"),
            optopt("a", "012345678901234567890123456789",
                     "Desc", "VAL"),
            optflag("k", "kiwi", "Desc"),
            optflagopt("p", "", "Desc", "VAL"),
            optmulti("l", "", "Desc", "VAL"),
        ];

        let expected = ~"Usage: fruits -b VAL [-a VAL] [-k] [-p [VAL]] [-l VAL]..";
        let generated_usage = short_usage("fruits", optgroups);

        debug!("expected: <<{}>>", expected);
        debug!("generated: <<{}>>", generated_usage);
        assert_eq!(generated_usage, expected);
    }
}
