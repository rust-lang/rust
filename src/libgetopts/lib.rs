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
//! fn do_work(inp: &str, out: Option<String>) {
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
//!     let args: Vec<String> = os::args().iter()
//!                                       .map(|x| x.to_string())
//!                                       .collect();
//!
//!     let program = args.get(0).clone();
//!
//!     let opts = [
//!         optopt("o", "", "set output file name", "NAME"),
//!         optflag("h", "help", "print this help menu")
//!     ];
//!     let matches = match getopts(args.tail(), opts) {
//!         Ok(m) => { m }
//!         Err(f) => { fail!(f.to_str()) }
//!     };
//!     if matches.opt_present("h") {
//!         print_usage(program.as_slice(), opts);
//!         return;
//!     }
//!     let output = matches.opt_str("o");
//!     let input = if !matches.free.is_empty() {
//!         (*matches.free.get(0)).clone()
//!     } else {
//!         print_usage(program.as_slice(), opts);
//!         return;
//!     };
//!     do_work(input.as_slice(), output);
//! }
//! ~~~

#![crate_id = "getopts#0.11.0-pre"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/",
       html_playground_url = "http://play.rust-lang.org/")]
#![feature(globs, phase)]
#![deny(missing_doc)]

#[cfg(test)] extern crate debug;
#[cfg(test)] #[phase(plugin, link)] extern crate log;

use std::cmp::PartialEq;
use std::fmt;
use std::result::{Err, Ok};
use std::result;
use std::string::String;

/// Name of an option. Either a string or a single char.
#[deriving(Clone, PartialEq, Eq)]
pub enum Name {
    /// A string representing the long name of an option.
    /// For example: "help"
    Long(String),
    /// A char representing the short name of an option.
    /// For example: 'h'
    Short(char),
}

/// Describes whether an option has an argument.
#[deriving(Clone, PartialEq, Eq)]
pub enum HasArg {
    /// The option requires an argument.
    Yes,
    /// The option is just a flag, therefore no argument.
    No,
    /// The option argument is optional and it could or not exist.
    Maybe,
}

/// Describes how often an option may occur.
#[deriving(Clone, PartialEq, Eq)]
pub enum Occur {
    /// The option occurs once.
    Req,
    /// The option could or not occur.
    Optional,
    /// The option occurs once or multiple times.
    Multi,
}

/// A description of a possible option.
#[deriving(Clone, PartialEq, Eq)]
pub struct Opt {
    /// Name of the option
    pub name: Name,
    /// Whether it has an argument
    pub hasarg: HasArg,
    /// How often it can occur
    pub occur: Occur,
    /// Which options it aliases
    pub aliases: Vec<Opt>,
}

/// One group of options, e.g., both -h and --help, along with
/// their shared description and properties.
#[deriving(Clone, PartialEq, Eq)]
pub struct OptGroup {
    /// Short Name of the `OptGroup`
    pub short_name: String,
    /// Long Name of the `OptGroup`
    pub long_name: String,
    /// Hint
    pub hint: String,
    /// Description
    pub desc: String,
    /// Whether it has an argument
    pub hasarg: HasArg,
    /// How often it can occur
    pub occur: Occur
}

/// Describes whether an option is given at all or has a value.
#[deriving(Clone, PartialEq, Eq)]
enum Optval {
    Val(String),
    Given,
}

/// The result of checking command line arguments. Contains a vector
/// of matches and a vector of free strings.
#[deriving(Clone, PartialEq, Eq)]
pub struct Matches {
    /// Options that matched
    opts: Vec<Opt>,
    /// Values of the Options that matched
    vals: Vec<Vec<Optval>>,
    /// Free string fragments
    pub free: Vec<String>,
}

/// The type returned when the command line does not conform to the
/// expected format. Use the `Show` implementation to output detailed
/// information.
#[deriving(Clone, PartialEq, Eq)]
pub enum Fail_ {
    /// The option requires an argument but none was passed.
    ArgumentMissing(String),
    /// The passed option is not declared among the possible options.
    UnrecognizedOption(String),
    /// A required option is not present.
    OptionMissing(String),
    /// A single occurrence option is being used multiple times.
    OptionDuplicated(String),
    /// There's an argument being passed to a non-argument option.
    UnexpectedArgument(String),
}

/// The type of failure that occurred.
#[deriving(PartialEq, Eq)]
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
            Long(nm.to_string())
        }
    }

    fn to_str(&self) -> String {
        match *self {
            Short(ch) => ch.to_str(),
            Long(ref s) => s.to_string()
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
                aliases: Vec::new()
            },
            (1,0) => Opt {
                name: Short(short_name.as_slice().char_at(0)),
                hasarg: hasarg,
                occur: occur,
                aliases: Vec::new()
            },
            (1,_) => Opt {
                name: Long((long_name)),
                hasarg: hasarg,
                occur:  occur,
                aliases: vec!(
                    Opt {
                        name: Short(short_name.as_slice().char_at(0)),
                        hasarg: hasarg,
                        occur:  occur,
                        aliases: Vec::new()
                    }
                )
            },
            (_,_) => fail!("something is wrong with the long-form opt")
        }
    }
}

impl Matches {
    fn opt_vals(&self, nm: &str) -> Vec<Optval> {
        match find_opt(self.opts.as_slice(), Name::from_str(nm)) {
            Some(id) => (*self.vals.get(id)).clone(),
            None => fail!("No option '{}' defined", nm)
        }
    }

    fn opt_val(&self, nm: &str) -> Option<Optval> {
        let vals = self.opt_vals(nm);
        if vals.is_empty() {
            None
        } else {
            Some((*vals.get(0)).clone())
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
    pub fn opts_present(&self, names: &[String]) -> bool {
        for nm in names.iter() {
            match find_opt(self.opts.as_slice(),
                           Name::from_str(nm.as_slice())) {
                Some(id) if !self.vals.get(id).is_empty() => return true,
                _ => (),
            };
        }
        false
    }

    /// Returns the string argument supplied to one of several matching options or `None`.
    pub fn opts_str(&self, names: &[String]) -> Option<String> {
        for nm in names.iter() {
            match self.opt_val(nm.as_slice()) {
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
    pub fn opt_strs(&self, nm: &str) -> Vec<String> {
        let mut acc: Vec<String> = Vec::new();
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
    pub fn opt_str(&self, nm: &str) -> Option<String> {
        let vals = self.opt_vals(nm);
        if vals.is_empty() {
            return None::<String>;
        }
        match vals.get(0) {
            &Val(ref s) => Some((*s).clone()),
            _ => None
        }
    }


    /// Returns the matching string, a default, or none.
    ///
    /// Returns none if the option was not present, `def` if the option was
    /// present but no argument was provided, and the argument if the option was
    /// present and an argument was provided.
    pub fn opt_default(&self, nm: &str, def: &str) -> Option<String> {
        let vals = self.opt_vals(nm);
        if vals.is_empty() {
            return None;
        }
        match vals.get(0) {
            &Val(ref s) => Some((*s).clone()),
            _ => Some(def.to_string())
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
        short_name: short_name.to_string(),
        long_name: long_name.to_string(),
        hint: hint.to_string(),
        desc: desc.to_string(),
        hasarg: Yes,
        occur: Req
    }
}

/// Create a long option that is optional and takes an argument.
pub fn optopt(short_name: &str, long_name: &str, desc: &str, hint: &str) -> OptGroup {
    let len = short_name.len();
    assert!(len == 1 || len == 0);
    OptGroup {
        short_name: short_name.to_string(),
        long_name: long_name.to_string(),
        hint: hint.to_string(),
        desc: desc.to_string(),
        hasarg: Yes,
        occur: Optional
    }
}

/// Create a long option that is optional and does not take an argument.
pub fn optflag(short_name: &str, long_name: &str, desc: &str) -> OptGroup {
    let len = short_name.len();
    assert!(len == 1 || len == 0);
    OptGroup {
        short_name: short_name.to_string(),
        long_name: long_name.to_string(),
        hint: "".to_string(),
        desc: desc.to_string(),
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
        short_name: short_name.to_string(),
        long_name: long_name.to_string(),
        hint: "".to_string(),
        desc: desc.to_string(),
        hasarg: No,
        occur: Multi
    }
}

/// Create a long option that is optional and takes an optional argument.
pub fn optflagopt(short_name: &str, long_name: &str, desc: &str, hint: &str) -> OptGroup {
    let len = short_name.len();
    assert!(len == 1 || len == 0);
    OptGroup {
        short_name: short_name.to_string(),
        long_name: long_name.to_string(),
        hint: hint.to_string(),
        desc: desc.to_string(),
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
        short_name: short_name.to_string(),
        long_name: long_name.to_string(),
        hint: hint.to_string(),
        desc: desc.to_string(),
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
        short_name: short_name.to_string(),
        long_name: long_name.to_string(),
        hint: hint.to_string(),
        desc: desc.to_string(),
        hasarg: hasarg,
        occur: occur
    }
}

impl Fail_ {
    /// Convert a `Fail_` enum into an error string.
    #[deprecated="use `Show` (`{}` format specifier)"]
    pub fn to_err_msg(self) -> String {
        self.to_str()
    }
}

impl fmt::Show for Fail_ {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ArgumentMissing(ref nm) => {
                write!(f, "Argument to option '{}' missing.", *nm)
            }
            UnrecognizedOption(ref nm) => {
                write!(f, "Unrecognized option: '{}'.", *nm)
            }
            OptionMissing(ref nm) => {
                write!(f, "Required option '{}' missing.", *nm)
            }
            OptionDuplicated(ref nm) => {
                write!(f, "Option '{}' given more than once.", *nm)
            }
            UnexpectedArgument(ref nm) => {
                write!(f, "Option '{}' does not take an argument.", *nm)
            }
        }
    }
}

/// Parse command line arguments according to the provided options.
///
/// On success returns `Ok(Opt)`. Use methods such as `opt_present`
/// `opt_str`, etc. to interrogate results.  Returns `Err(Fail_)` on
/// failure: use the `Show` implementation of `Fail_` to display
/// information about it.
pub fn getopts(args: &[String], optgrps: &[OptGroup]) -> Result {
    let opts: Vec<Opt> = optgrps.iter().map(|x| x.long_to_short()).collect();
    let n_opts = opts.len();

    fn f(_x: uint) -> Vec<Optval> { return Vec::new(); }

    let mut vals = Vec::from_fn(n_opts, f);
    let mut free: Vec<String> = Vec::new();
    let l = args.len();
    let mut i = 0;
    while i < l {
        let cur = args[i].clone();
        let curlen = cur.len();
        if !is_arg(cur.as_slice()) {
            free.push(cur);
        } else if cur.as_slice() == "--" {
            let mut j = i + 1;
            while j < l { free.push(args[j].clone()); j += 1; }
            break;
        } else {
            let mut names;
            let mut i_arg = None;
            if cur.as_slice()[1] == '-' as u8 {
                let tail = cur.as_slice().slice(2, curlen);
                let tail_eq: Vec<&str> = tail.split('=').collect();
                if tail_eq.len() <= 1 {
                    names = vec!(Long(tail.to_string()));
                } else {
                    names =
                        vec!(Long((*tail_eq.get(0)).to_string()));
                    i_arg = Some((*tail_eq.get(1)).to_string());
                }
            } else {
                let mut j = 1;
                let mut last_valid_opt_id = None;
                names = Vec::new();
                while j < curlen {
                    let range = cur.as_slice().char_range_at(j);
                    let opt = Short(range.ch);

                    /* In a series of potential options (eg. -aheJ), if we
                       see one which takes an argument, we assume all
                       subsequent characters make up the argument. This
                       allows options such as -L/usr/local/lib/foo to be
                       interpreted correctly
                    */

                    match find_opt(opts.as_slice(), opt.clone()) {
                      Some(id) => last_valid_opt_id = Some(id),
                      None => {
                        let arg_follows =
                            last_valid_opt_id.is_some() &&
                            match opts.get(last_valid_opt_id.unwrap())
                              .hasarg {

                              Yes | Maybe => true,
                              No => false
                            };
                        if arg_follows && j < curlen {
                            i_arg = Some(cur.as_slice()
                                            .slice(j, curlen).to_string());
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
                let optid = match find_opt(opts.as_slice(), (*nm).clone()) {
                  Some(id) => id,
                  None => return Err(UnrecognizedOption(nm.to_str()))
                };
                match opts.get(optid).hasarg {
                  No => {
                    if !i_arg.is_none() {
                        return Err(UnexpectedArgument(nm.to_str()));
                    }
                    vals.get_mut(optid).push(Given);
                  }
                  Maybe => {
                    if !i_arg.is_none() {
                        vals.get_mut(optid)
                            .push(Val((i_arg.clone())
                            .unwrap()));
                    } else if name_pos < names.len() || i + 1 == l ||
                            is_arg(args[i + 1].as_slice()) {
                        vals.get_mut(optid).push(Given);
                    } else {
                        i += 1;
                        vals.get_mut(optid).push(Val(args[i].clone()));
                    }
                  }
                  Yes => {
                    if !i_arg.is_none() {
                        vals.get_mut(optid).push(Val(i_arg.clone().unwrap()));
                    } else if i + 1 == l {
                        return Err(ArgumentMissing(nm.to_str()));
                    } else {
                        i += 1;
                        vals.get_mut(optid).push(Val(args[i].clone()));
                    }
                  }
                }
            }
        }
        i += 1;
    }
    i = 0u;
    while i < n_opts {
        let n = vals.get(i).len();
        let occ = opts.get(i).occur;
        if occ == Req {
            if n == 0 {
                return Err(OptionMissing(opts.get(i).name.to_str()));
            }
        }
        if occ != Multi {
            if n > 1 {
                return Err(OptionDuplicated(opts.get(i).name.to_str()));
            }
        }
        i += 1;
    }
    Ok(Matches {
        opts: opts,
        vals: vals,
        free: free
    })
}

/// Derive a usage message from a set of long options.
pub fn usage(brief: &str, opts: &[OptGroup]) -> String {

    let desc_sep = format!("\n{}", " ".repeat(24));

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
                row.push_str(short_name.as_slice());
                row.push_char(' ');
            }
            _ => fail!("the short name should only be 1 ascii char long"),
        }

        // long option
        match long_name.len() {
            0 => {}
            _ => {
                row.push_str("--");
                row.push_str(long_name.as_slice());
                row.push_char(' ');
            }
        }

        // arg
        match hasarg {
            No => {}
            Yes => row.push_str(hint.as_slice()),
            Maybe => {
                row.push_char('[');
                row.push_str(hint.as_slice());
                row.push_char(']');
            }
        }

        // FIXME: #5516 should be graphemes not codepoints
        // here we just need to indent the start of the description
        let rowlen = row.as_slice().char_len();
        if rowlen < 24 {
            for _ in range(0, 24 - rowlen) {
                row.push_char(' ');
            }
        } else {
            row.push_str(desc_sep.as_slice())
        }

        // Normalize desc to contain words separated by one space character
        let mut desc_normalized_whitespace = String::new();
        for word in desc.as_slice().words() {
            desc_normalized_whitespace.push_str(word);
            desc_normalized_whitespace.push_char(' ');
        }

        // FIXME: #5516 should be graphemes not codepoints
        let mut desc_rows = Vec::new();
        each_split_within(desc_normalized_whitespace.as_slice(),
                          54,
                          |substr| {
            desc_rows.push(substr.to_string());
            true
        });

        // FIXME: #5516 should be graphemes not codepoints
        // wrapped description
        row.push_str(desc_rows.connect(desc_sep.as_slice()).as_slice());

        row
    });

    format!("{}\n\nOptions:\n{}\n", brief,
            rows.collect::<Vec<String>>().connect("\n"))
}

fn format_option(opt: &OptGroup) -> String {
    let mut line = String::new();

    if opt.occur != Req {
        line.push_char('[');
    }

    // Use short_name is possible, but fallback to long_name.
    if opt.short_name.len() > 0 {
        line.push_char('-');
        line.push_str(opt.short_name.as_slice());
    } else {
        line.push_str("--");
        line.push_str(opt.long_name.as_slice());
    }

    if opt.hasarg != No {
        line.push_char(' ');
        if opt.hasarg == Maybe {
            line.push_char('[');
        }
        line.push_str(opt.hint.as_slice());
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
pub fn short_usage(program_name: &str, opts: &[OptGroup]) -> String {
    let mut line = format!("Usage: {} ", program_name);
    line.push_str(opts.iter()
                      .map(format_option)
                      .collect::<Vec<String>>()
                      .connect(" ")
                      .as_slice());
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
    fn t(s: &str, i: uint, u: &[String]) {
        let mut v = Vec::new();
        each_split_within(s, i, |s| { v.push(s.to_string()); true });
        assert!(v.iter().zip(u.iter()).all(|(a,b)| a == b));
    }
    t("", 0, []);
    t("", 15, []);
    t("hello", 15, ["hello".to_string()]);
    t("\nMary had a little lamb\nLittle lamb\n", 15, [
        "Mary had a".to_string(),
        "little lamb".to_string(),
        "Little lamb".to_string()
    ]);
    t("\nMary had a little lamb\nLittle lamb\n", ::std::uint::MAX,
        ["Mary had a little lamb\nLittle lamb".to_string()]);
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
        let long_args = vec!("--test=20".to_string());
        let opts = vec!(reqopt("t", "test", "testing", "TEST"));
        let rs = getopts(long_args.as_slice(), opts.as_slice());
        match rs {
          Ok(ref m) => {
            assert!(m.opt_present("test"));
            assert_eq!(m.opt_str("test").unwrap(), "20".to_string());
            assert!(m.opt_present("t"));
            assert_eq!(m.opt_str("t").unwrap(), "20".to_string());
          }
          _ => { fail!("test_reqopt failed (long arg)"); }
        }
        let short_args = vec!("-t".to_string(), "20".to_string());
        match getopts(short_args.as_slice(), opts.as_slice()) {
          Ok(ref m) => {
            assert!((m.opt_present("test")));
            assert_eq!(m.opt_str("test").unwrap(), "20".to_string());
            assert!((m.opt_present("t")));
            assert_eq!(m.opt_str("t").unwrap(), "20".to_string());
          }
          _ => { fail!("test_reqopt failed (short arg)"); }
        }
    }

    #[test]
    fn test_reqopt_missing() {
        let args = vec!("blah".to_string());
        let opts = vec!(reqopt("t", "test", "testing", "TEST"));
        let rs = getopts(args.as_slice(), opts.as_slice());
        match rs {
          Err(f) => check_fail_type(f, OptionMissing_),
          _ => fail!()
        }
    }

    #[test]
    fn test_reqopt_no_arg() {
        let long_args = vec!("--test".to_string());
        let opts = vec!(reqopt("t", "test", "testing", "TEST"));
        let rs = getopts(long_args.as_slice(), opts.as_slice());
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
        let short_args = vec!("-t".to_string());
        match getopts(short_args.as_slice(), opts.as_slice()) {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
    }

    #[test]
    fn test_reqopt_multi() {
        let args = vec!("--test=20".to_string(), "-t".to_string(), "30".to_string());
        let opts = vec!(reqopt("t", "test", "testing", "TEST"));
        let rs = getopts(args.as_slice(), opts.as_slice());
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail!()
        }
    }

    // Tests for optopt
    #[test]
    fn test_optopt() {
        let long_args = vec!("--test=20".to_string());
        let opts = vec!(optopt("t", "test", "testing", "TEST"));
        let rs = getopts(long_args.as_slice(), opts.as_slice());
        match rs {
          Ok(ref m) => {
            assert!(m.opt_present("test"));
            assert_eq!(m.opt_str("test").unwrap(), "20".to_string());
            assert!((m.opt_present("t")));
            assert_eq!(m.opt_str("t").unwrap(), "20".to_string());
          }
          _ => fail!()
        }
        let short_args = vec!("-t".to_string(), "20".to_string());
        match getopts(short_args.as_slice(), opts.as_slice()) {
          Ok(ref m) => {
            assert!((m.opt_present("test")));
            assert_eq!(m.opt_str("test").unwrap(), "20".to_string());
            assert!((m.opt_present("t")));
            assert_eq!(m.opt_str("t").unwrap(), "20".to_string());
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optopt_missing() {
        let args = vec!("blah".to_string());
        let opts = vec!(optopt("t", "test", "testing", "TEST"));
        let rs = getopts(args.as_slice(), opts.as_slice());
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
        let long_args = vec!("--test".to_string());
        let opts = vec!(optopt("t", "test", "testing", "TEST"));
        let rs = getopts(long_args.as_slice(), opts.as_slice());
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
        let short_args = vec!("-t".to_string());
        match getopts(short_args.as_slice(), opts.as_slice()) {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
    }

    #[test]
    fn test_optopt_multi() {
        let args = vec!("--test=20".to_string(), "-t".to_string(), "30".to_string());
        let opts = vec!(optopt("t", "test", "testing", "TEST"));
        let rs = getopts(args.as_slice(), opts.as_slice());
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail!()
        }
    }

    // Tests for optflag
    #[test]
    fn test_optflag() {
        let long_args = vec!("--test".to_string());
        let opts = vec!(optflag("t", "test", "testing"));
        let rs = getopts(long_args.as_slice(), opts.as_slice());
        match rs {
          Ok(ref m) => {
            assert!(m.opt_present("test"));
            assert!(m.opt_present("t"));
          }
          _ => fail!()
        }
        let short_args = vec!("-t".to_string());
        match getopts(short_args.as_slice(), opts.as_slice()) {
          Ok(ref m) => {
            assert!(m.opt_present("test"));
            assert!(m.opt_present("t"));
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflag_missing() {
        let args = vec!("blah".to_string());
        let opts = vec!(optflag("t", "test", "testing"));
        let rs = getopts(args.as_slice(), opts.as_slice());
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
        let args = vec!("--test=20".to_string());
        let opts = vec!(optflag("t", "test", "testing"));
        let rs = getopts(args.as_slice(), opts.as_slice());
        match rs {
          Err(f) => {
            check_fail_type(f, UnexpectedArgument_);
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflag_multi() {
        let args = vec!("--test".to_string(), "-t".to_string());
        let opts = vec!(optflag("t", "test", "testing"));
        let rs = getopts(args.as_slice(), opts.as_slice());
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail!()
        }
    }

    #[test]
    fn test_optflag_short_arg() {
        let args = vec!("-t".to_string(), "20".to_string());
        let opts = vec!(optflag("t", "test", "testing"));
        let rs = getopts(args.as_slice(), opts.as_slice());
        match rs {
          Ok(ref m) => {
            // The next variable after the flag is just a free argument

            assert!(*m.free.get(0) == "20".to_string());
          }
          _ => fail!()
        }
    }

    // Tests for optflagmulti
    #[test]
    fn test_optflagmulti_short1() {
        let args = vec!("-v".to_string());
        let opts = vec!(optflagmulti("v", "verbose", "verbosity"));
        let rs = getopts(args.as_slice(), opts.as_slice());
        match rs {
          Ok(ref m) => {
            assert_eq!(m.opt_count("v"), 1);
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflagmulti_short2a() {
        let args = vec!("-v".to_string(), "-v".to_string());
        let opts = vec!(optflagmulti("v", "verbose", "verbosity"));
        let rs = getopts(args.as_slice(), opts.as_slice());
        match rs {
          Ok(ref m) => {
            assert_eq!(m.opt_count("v"), 2);
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflagmulti_short2b() {
        let args = vec!("-vv".to_string());
        let opts = vec!(optflagmulti("v", "verbose", "verbosity"));
        let rs = getopts(args.as_slice(), opts.as_slice());
        match rs {
          Ok(ref m) => {
            assert_eq!(m.opt_count("v"), 2);
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflagmulti_long1() {
        let args = vec!("--verbose".to_string());
        let opts = vec!(optflagmulti("v", "verbose", "verbosity"));
        let rs = getopts(args.as_slice(), opts.as_slice());
        match rs {
          Ok(ref m) => {
            assert_eq!(m.opt_count("verbose"), 1);
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflagmulti_long2() {
        let args = vec!("--verbose".to_string(), "--verbose".to_string());
        let opts = vec!(optflagmulti("v", "verbose", "verbosity"));
        let rs = getopts(args.as_slice(), opts.as_slice());
        match rs {
          Ok(ref m) => {
            assert_eq!(m.opt_count("verbose"), 2);
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflagmulti_mix() {
        let args = vec!("--verbose".to_string(), "-v".to_string(),
                        "-vv".to_string(), "verbose".to_string());
        let opts = vec!(optflagmulti("v", "verbose", "verbosity"));
        let rs = getopts(args.as_slice(), opts.as_slice());
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
        let long_args = vec!("--test=20".to_string());
        let opts = vec!(optmulti("t", "test", "testing", "TEST"));
        let rs = getopts(long_args.as_slice(), opts.as_slice());
        match rs {
          Ok(ref m) => {
            assert!((m.opt_present("test")));
            assert_eq!(m.opt_str("test").unwrap(), "20".to_string());
            assert!((m.opt_present("t")));
            assert_eq!(m.opt_str("t").unwrap(), "20".to_string());
          }
          _ => fail!()
        }
        let short_args = vec!("-t".to_string(), "20".to_string());
        match getopts(short_args.as_slice(), opts.as_slice()) {
          Ok(ref m) => {
            assert!((m.opt_present("test")));
            assert_eq!(m.opt_str("test").unwrap(), "20".to_string());
            assert!((m.opt_present("t")));
            assert_eq!(m.opt_str("t").unwrap(), "20".to_string());
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optmulti_missing() {
        let args = vec!("blah".to_string());
        let opts = vec!(optmulti("t", "test", "testing", "TEST"));
        let rs = getopts(args.as_slice(), opts.as_slice());
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
        let long_args = vec!("--test".to_string());
        let opts = vec!(optmulti("t", "test", "testing", "TEST"));
        let rs = getopts(long_args.as_slice(), opts.as_slice());
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
        let short_args = vec!("-t".to_string());
        match getopts(short_args.as_slice(), opts.as_slice()) {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
    }

    #[test]
    fn test_optmulti_multi() {
        let args = vec!("--test=20".to_string(), "-t".to_string(), "30".to_string());
        let opts = vec!(optmulti("t", "test", "testing", "TEST"));
        let rs = getopts(args.as_slice(), opts.as_slice());
        match rs {
          Ok(ref m) => {
              assert!(m.opt_present("test"));
              assert_eq!(m.opt_str("test").unwrap(), "20".to_string());
              assert!(m.opt_present("t"));
              assert_eq!(m.opt_str("t").unwrap(), "20".to_string());
              let pair = m.opt_strs("test");
              assert!(*pair.get(0) == "20".to_string());
              assert!(*pair.get(1) == "30".to_string());
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_unrecognized_option() {
        let long_args = vec!("--untest".to_string());
        let opts = vec!(optmulti("t", "test", "testing", "TEST"));
        let rs = getopts(long_args.as_slice(), opts.as_slice());
        match rs {
          Err(f) => check_fail_type(f, UnrecognizedOption_),
          _ => fail!()
        }
        let short_args = vec!("-u".to_string());
        match getopts(short_args.as_slice(), opts.as_slice()) {
          Err(f) => check_fail_type(f, UnrecognizedOption_),
          _ => fail!()
        }
    }

    #[test]
    fn test_combined() {
        let args =
            vec!("prog".to_string(),
                 "free1".to_string(),
                 "-s".to_string(),
                 "20".to_string(),
                 "free2".to_string(),
                 "--flag".to_string(),
                 "--long=30".to_string(),
                 "-f".to_string(),
                 "-m".to_string(),
                 "40".to_string(),
                 "-m".to_string(),
                 "50".to_string(),
                 "-n".to_string(),
                 "-A B".to_string(),
                 "-n".to_string(),
                 "-60 70".to_string());
        let opts =
            vec!(optopt("s", "something", "something", "SOMETHING"),
              optflag("", "flag", "a flag"),
              reqopt("", "long", "hi", "LONG"),
              optflag("f", "", "another flag"),
              optmulti("m", "", "mmmmmm", "YUM"),
              optmulti("n", "", "nothing", "NOTHING"),
              optopt("", "notpresent", "nothing to see here", "NOPE"));
        let rs = getopts(args.as_slice(), opts.as_slice());
        match rs {
          Ok(ref m) => {
            assert!(*m.free.get(0) == "prog".to_string());
            assert!(*m.free.get(1) == "free1".to_string());
            assert_eq!(m.opt_str("s").unwrap(), "20".to_string());
            assert!(*m.free.get(2) == "free2".to_string());
            assert!((m.opt_present("flag")));
            assert_eq!(m.opt_str("long").unwrap(), "30".to_string());
            assert!((m.opt_present("f")));
            let pair = m.opt_strs("m");
            assert!(*pair.get(0) == "40".to_string());
            assert!(*pair.get(1) == "50".to_string());
            let pair = m.opt_strs("n");
            assert!(*pair.get(0) == "-A B".to_string());
            assert!(*pair.get(1) == "-60 70".to_string());
            assert!((!m.opt_present("notpresent")));
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_multi() {
        let opts = vec!(optopt("e", "", "encrypt", "ENCRYPT"),
                     optopt("", "encrypt", "encrypt", "ENCRYPT"),
                     optopt("f", "", "flag", "FLAG"));

        let args_single = vec!("-e".to_string(), "foo".to_string());
        let matches_single = &match getopts(args_single.as_slice(),
                                            opts.as_slice()) {
          result::Ok(m) => m,
          result::Err(_) => fail!()
        };
        assert!(matches_single.opts_present(["e".to_string()]));
        assert!(matches_single.opts_present(["encrypt".to_string(), "e".to_string()]));
        assert!(matches_single.opts_present(["e".to_string(), "encrypt".to_string()]));
        assert!(!matches_single.opts_present(["encrypt".to_string()]));
        assert!(!matches_single.opts_present(["thing".to_string()]));
        assert!(!matches_single.opts_present([]));

        assert_eq!(matches_single.opts_str(["e".to_string()]).unwrap(), "foo".to_string());
        assert_eq!(matches_single.opts_str(["e".to_string(), "encrypt".to_string()]).unwrap(),
                   "foo".to_string());
        assert_eq!(matches_single.opts_str(["encrypt".to_string(), "e".to_string()]).unwrap(),
                   "foo".to_string());

        let args_both = vec!("-e".to_string(), "foo".to_string(), "--encrypt".to_string(),
                             "foo".to_string());
        let matches_both = &match getopts(args_both.as_slice(),
                                          opts.as_slice()) {
          result::Ok(m) => m,
          result::Err(_) => fail!()
        };
        assert!(matches_both.opts_present(["e".to_string()]));
        assert!(matches_both.opts_present(["encrypt".to_string()]));
        assert!(matches_both.opts_present(["encrypt".to_string(), "e".to_string()]));
        assert!(matches_both.opts_present(["e".to_string(), "encrypt".to_string()]));
        assert!(!matches_both.opts_present(["f".to_string()]));
        assert!(!matches_both.opts_present(["thing".to_string()]));
        assert!(!matches_both.opts_present([]));

        assert_eq!(matches_both.opts_str(["e".to_string()]).unwrap(), "foo".to_string());
        assert_eq!(matches_both.opts_str(["encrypt".to_string()]).unwrap(), "foo".to_string());
        assert_eq!(matches_both.opts_str(["e".to_string(), "encrypt".to_string()]).unwrap(),
                   "foo".to_string());
        assert_eq!(matches_both.opts_str(["encrypt".to_string(), "e".to_string()]).unwrap(),
                   "foo".to_string());
    }

    #[test]
    fn test_nospace() {
        let args = vec!("-Lfoo".to_string(), "-M.".to_string());
        let opts = vec!(optmulti("L", "", "library directory", "LIB"),
                     optmulti("M", "", "something", "MMMM"));
        let matches = &match getopts(args.as_slice(), opts.as_slice()) {
          result::Ok(m) => m,
          result::Err(_) => fail!()
        };
        assert!(matches.opts_present(["L".to_string()]));
        assert_eq!(matches.opts_str(["L".to_string()]).unwrap(), "foo".to_string());
        assert!(matches.opts_present(["M".to_string()]));
        assert_eq!(matches.opts_str(["M".to_string()]).unwrap(), ".".to_string());

    }

    #[test]
    fn test_long_to_short() {
        let mut short = Opt {
            name: Long("banana".to_string()),
            hasarg: Yes,
            occur: Req,
            aliases: Vec::new(),
        };
        short.aliases = vec!(Opt { name: Short('b'),
                                hasarg: Yes,
                                occur: Req,
                                aliases: Vec::new() });
        let verbose = reqopt("b", "banana", "some bananas", "VAL");

        assert!(verbose.long_to_short() == short);
    }

    #[test]
    fn test_aliases_long_and_short() {
        let opts = vec!(
            optflagmulti("a", "apple", "Desc"));

        let args = vec!("-a".to_string(), "--apple".to_string(), "-a".to_string());

        let matches = getopts(args.as_slice(), opts.as_slice()).unwrap();
        assert_eq!(3, matches.opt_count("a"));
        assert_eq!(3, matches.opt_count("apple"));
    }

    #[test]
    fn test_usage() {
        let optgroups = vec!(
            reqopt("b", "banana", "Desc", "VAL"),
            optopt("a", "012345678901234567890123456789",
                             "Desc", "VAL"),
            optflag("k", "kiwi", "Desc"),
            optflagopt("p", "", "Desc", "VAL"),
            optmulti("l", "", "Desc", "VAL"));

        let expected =
"Usage: fruits

Options:
    -b --banana VAL     Desc
    -a --012345678901234567890123456789 VAL
                        Desc
    -k --kiwi           Desc
    -p [VAL]            Desc
    -l VAL              Desc
".to_string();

        let generated_usage = usage("Usage: fruits", optgroups.as_slice());

        debug!("expected: <<{}>>", expected);
        debug!("generated: <<{}>>", generated_usage);
        assert_eq!(generated_usage, expected);
    }

    #[test]
    fn test_usage_description_wrapping() {
        // indentation should be 24 spaces
        // lines wrap after 78: or rather descriptions wrap after 54

        let optgroups = vec!(
            optflag("k", "kiwi",
                "This is a long description which won't be wrapped..+.."), // 54
            optflag("a", "apple",
                "This is a long description which _will_ be wrapped..+.."));

        let expected =
"Usage: fruits

Options:
    -k --kiwi           This is a long description which won't be wrapped..+..
    -a --apple          This is a long description which _will_ be
                        wrapped..+..
".to_string();

        let usage = usage("Usage: fruits", optgroups.as_slice());

        debug!("expected: <<{}>>", expected);
        debug!("generated: <<{}>>", usage);
        assert!(usage == expected)
    }

    #[test]
    fn test_usage_description_multibyte_handling() {
        let optgroups = vec!(
            optflag("k", "k\u2013w\u2013",
                "The word kiwi is normally spelled with two i's"),
            optflag("a", "apple",
                "This \u201Cdescription\u201D has some characters that could \
confuse the line wrapping; an apple costs 0.51€ in some parts of Europe."));

        let expected =
"Usage: fruits

Options:
    -k --k–w–           The word kiwi is normally spelled with two i's
    -a --apple          This “description” has some characters that could
                        confuse the line wrapping; an apple costs 0.51€ in
                        some parts of Europe.
".to_string();

        let usage = usage("Usage: fruits", optgroups.as_slice());

        debug!("expected: <<{}>>", expected);
        debug!("generated: <<{}>>", usage);
        assert!(usage == expected)
    }

    #[test]
    fn test_short_usage() {
        let optgroups = vec!(
            reqopt("b", "banana", "Desc", "VAL"),
            optopt("a", "012345678901234567890123456789",
                     "Desc", "VAL"),
            optflag("k", "kiwi", "Desc"),
            optflagopt("p", "", "Desc", "VAL"),
            optmulti("l", "", "Desc", "VAL"));

        let expected = "Usage: fruits -b VAL [-a VAL] [-k] [-p [VAL]] [-l VAL]..".to_string();
        let generated_usage = short_usage("fruits", optgroups.as_slice());

        debug!("expected: <<{}>>", expected);
        debug!("generated: <<{}>>", generated_usage);
        assert_eq!(generated_usage, expected);
    }
}
