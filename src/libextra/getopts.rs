// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Simple getopt alternative.
 *
 * Construct a vector of options, either by using reqopt, optopt, and optflag
 * or by building them from components yourself, and pass them to getopts,
 * along with a vector of actual arguments (not including argv[0]). You'll
 * either get a failure code back, or a match. You'll have to verify whether
 * the amount of 'free' arguments in the match is what you expect. Use opt_*
 * accessors to get argument values out of the matches object.
 *
 * Single-character options are expected to appear on the command line with a
 * single preceding dash; multiple-character options are expected to be
 * proceeded by two dashes. Options that expect an argument accept their
 * argument following either a space or an equals sign. Single-character
 * options don't require the space.
 *
 * # Example
 *
 * The following example shows simple command line parsing for an application
 * that requires an input file to be specified, accepts an optional output
 * file name following -o, and accepts both -h and --help as optional flags.
 *
 * ```
 *    extern mod extra;
 *    use extra::getopts::*;
 *    use std::os;
 *
 *    fn do_work(in: &str, out: Option<~str>) {
 *        println(in);
 *        println(match out {
 *            Some(x) => x,
 *            None => ~"No Output"
 *        });
 *    }
 *
 *    fn print_usage(program: &str, _opts: &[Opt]) {
 *        printfln!("Usage: %s [options]", program);
 *        println("-o\t\tOutput");
 *        println("-h --help\tUsage");
 *    }
 *
 *    fn main() {
 *        let args = os::args();
 *
 *        let program = args[0].clone();
 *
 *        let opts = ~[
 *            optopt("o"),
 *            optflag("h"),
 *            optflag("help")
 *        ];
 *        let matches = match getopts(args.tail(), opts) {
 *            Ok(m) => { m }
 *            Err(f) => { fail!(fail_str(f)) }
 *        };
 *        if opt_present(&matches, "h") || opt_present(&matches, "help") {
 *            print_usage(program, opts);
 *            return;
 *        }
 *        let output = opt_maybe_str(&matches, "o");
 *        let input: &str = if !matches.free.is_empty() {
 *            matches.free[0].clone()
 *        } else {
 *            print_usage(program, opts);
 *            return;
 *        };
 *        do_work(input, output);
 *    }
 * ```
 */

#[allow(missing_doc)];


use std::cmp::Eq;
use std::result::{Err, Ok};
use std::result;
use std::option::{Some, None};
use std::str;
use std::vec;

#[deriving(Clone, Eq)]
pub enum Name {
    Long(~str),
    Short(char),
}

#[deriving(Clone, Eq)]
pub enum HasArg {
    Yes,
    No,
    Maybe,
}

#[deriving(Clone, Eq)]
pub enum Occur {
    Req,
    Optional,
    Multi,
}

/// A description of a possible option
#[deriving(Clone, Eq)]
pub struct Opt {
    name: Name,
    hasarg: HasArg,
    occur: Occur
}

fn mkname(nm: &str) -> Name {
  if nm.len() == 1u {
      Short(nm.char_at(0u))
  } else {
      Long(nm.to_owned())
  }
}

/// Create an option that is required and takes an argument
pub fn reqopt(name: &str) -> Opt {
    return Opt {name: mkname(name), hasarg: Yes, occur: Req};
}

/// Create an option that is optional and takes an argument
pub fn optopt(name: &str) -> Opt {
    return Opt {name: mkname(name), hasarg: Yes, occur: Optional};
}

/// Create an option that is optional and does not take an argument
pub fn optflag(name: &str) -> Opt {
    return Opt {name: mkname(name), hasarg: No, occur: Optional};
}

/** Create an option that is optional, does not take an argument,
  * and may occur multiple times.
  */
pub fn optflagmulti(name: &str) -> Opt {
    return Opt {name: mkname(name), hasarg: No, occur: Multi};
}

/// Create an option that is optional and takes an optional argument
pub fn optflagopt(name: &str) -> Opt {
    return Opt {name: mkname(name), hasarg: Maybe, occur: Optional};
}

/**
 * Create an option that is optional, takes an argument, and may occur
 * multiple times
 */
pub fn optmulti(name: &str) -> Opt {
    return Opt {name: mkname(name), hasarg: Yes, occur: Multi};
}

#[deriving(Clone, Eq)]
enum Optval {
    Val(~str),
    Given,
}

/**
 * The result of checking command line arguments. Contains a vector
 * of matches and a vector of free strings.
 */
#[deriving(Clone, Eq)]
pub struct Matches {
    opts: ~[Opt],
    vals: ~[~[Optval]],
    free: ~[~str]
}

fn is_arg(arg: &str) -> bool {
    return arg.len() > 1 && arg[0] == '-' as u8;
}

fn name_str(nm: &Name) -> ~str {
    return match *nm {
      Short(ch) => str::from_char(ch),
      Long(ref s) => (*s).clone()
    };
}

fn find_opt(opts: &[Opt], nm: Name) -> Option<uint> {
    opts.iter().position(|opt| opt.name == nm)
}

/**
 * The type returned when the command line does not conform to the
 * expected format. Pass this value to <fail_str> to get an error message.
 */
#[deriving(Clone, Eq)]
pub enum Fail_ {
    ArgumentMissing(~str),
    UnrecognizedOption(~str),
    OptionMissing(~str),
    OptionDuplicated(~str),
    UnexpectedArgument(~str),
}

/// Convert a `fail_` enum into an error string
pub fn fail_str(f: Fail_) -> ~str {
    return match f {
        ArgumentMissing(ref nm) => {
            fmt!("Argument to option '%s' missing.", *nm)
        }
        UnrecognizedOption(ref nm) => {
            fmt!("Unrecognized option: '%s'.", *nm)
        }
        OptionMissing(ref nm) => {
            fmt!("Required option '%s' missing.", *nm)
        }
        OptionDuplicated(ref nm) => {
            fmt!("Option '%s' given more than once.", *nm)
        }
        UnexpectedArgument(ref nm) => {
            fmt!("Option '%s' does not take an argument.", *nm)
        }
    };
}

/**
 * The result of parsing a command line with a set of options
 * (result::t<Matches, Fail_>)
 */
pub type Result = result::Result<Matches, Fail_>;

/**
 * Parse command line arguments according to the provided options
 *
 * On success returns `ok(Opt)`. Use functions such as `opt_present`
 * `opt_str`, etc. to interrogate results.  Returns `err(Fail_)` on failure.
 * Use <fail_str> to get an error message.
 */
pub fn getopts(args: &[~str], opts: &[Opt]) -> Result {
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
                let tail_eq: ~[&str] = tail.split_iter('=').collect();
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
                            match opts[last_valid_opt_id.get()]
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
            foreach nm in names.iter() {
                name_pos += 1;
                let optid = match find_opt(opts, (*nm).clone()) {
                  Some(id) => id,
                  None => return Err(UnrecognizedOption(name_str(nm)))
                };
                match opts[optid].hasarg {
                  No => {
                    if !i_arg.is_none() {
                        return Err(UnexpectedArgument(name_str(nm)));
                    }
                    vals[optid].push(Given);
                  }
                  Maybe => {
                    if !i_arg.is_none() {
                        vals[optid].push(Val((i_arg.clone()).get()));
                    } else if name_pos < names.len() ||
                                  i + 1 == l || is_arg(args[i + 1]) {
                        vals[optid].push(Given);
                    } else { i += 1; vals[optid].push(Val(args[i].clone())); }
                  }
                  Yes => {
                    if !i_arg.is_none() {
                        vals[optid].push(Val(i_arg.clone().get()));
                    } else if i + 1 == l {
                        return Err(ArgumentMissing(name_str(nm)));
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
                return Err(OptionMissing(name_str(&(opts[i].name))));
            }
        }
        if occ != Multi {
            if n > 1 {
                return Err(OptionDuplicated(name_str(&(opts[i].name))));
            }
        }
        i += 1;
    }
    return Ok(Matches {opts: opts.to_owned(),
               vals: vals,
               free: free});
}

fn opt_vals(mm: &Matches, nm: &str) -> ~[Optval] {
    return match find_opt(mm.opts, mkname(nm)) {
      Some(id) => mm.vals[id].clone(),
      None => {
        error!("No option '%s' defined", nm);
        fail!()
      }
    };
}

fn opt_val(mm: &Matches, nm: &str) -> Option<Optval> {
    let vals = opt_vals(mm, nm);
    if (vals.is_empty()) {
        None
    } else {
        Some(opt_vals(mm, nm)[0].clone())
    }
}

/// Returns true if an option was matched
pub fn opt_present(mm: &Matches, nm: &str) -> bool {
    !opt_vals(mm, nm).is_empty()
}

/// Returns the number of times an option was matched
pub fn opt_count(mm: &Matches, nm: &str) -> uint {
    opt_vals(mm, nm).len()
}

/// Returns true if any of several options were matched
pub fn opts_present(mm: &Matches, names: &[~str]) -> bool {
    foreach nm in names.iter() {
        match find_opt(mm.opts, mkname(*nm)) {
            Some(id) if !mm.vals[id].is_empty() => return true,
            _ => (),
        };
    }
    false
}


/**
 * Returns the string argument supplied to a matching option
 *
 * Fails if the option was not matched or if the match did not take an
 * argument
 */
pub fn opt_str(mm: &Matches, nm: &str) -> ~str {
    return match opt_val(mm, nm) {
        Some(Val(s)) => s,
        _ => fail!()
    };
}

/**
 * Returns the string argument supplied to one of several matching options
 *
 * Fails if the no option was provided from the given list, or if the no such
 * option took an argument
 */
pub fn opts_str(mm: &Matches, names: &[~str]) -> ~str {
    foreach nm in names.iter() {
        match opt_val(mm, *nm) {
          Some(Val(ref s)) => return (*s).clone(),
          _ => ()
        }
    }
    fail!();
}


/**
 * Returns a vector of the arguments provided to all matches of the given
 * option.
 *
 * Used when an option accepts multiple values.
 */
pub fn opt_strs(mm: &Matches, nm: &str) -> ~[~str] {
    let mut acc: ~[~str] = ~[];
    let r = opt_vals(mm, nm);
    foreach v in r.iter() {
        match *v { Val(ref s) => acc.push((*s).clone()), _ => () }
    }
    acc
}

/// Returns the string argument supplied to a matching option or none
pub fn opt_maybe_str(mm: &Matches, nm: &str) -> Option<~str> {
    let vals = opt_vals(mm, nm);
    if vals.is_empty() { return None::<~str>; }
    return match vals[0] {
        Val(ref s) => Some((*s).clone()),
        _ => None
    };
}


/**
 * Returns the matching string, a default, or none
 *
 * Returns none if the option was not present, `def` if the option was
 * present but no argument was provided, and the argument if the option was
 * present and an argument was provided.
 */
pub fn opt_default(mm: &Matches, nm: &str, def: &str) -> Option<~str> {
    let vals = opt_vals(mm, nm);
    if vals.is_empty() { return None::<~str>; }
    return match vals[0] { Val(ref s) => Some::<~str>((*s).clone()),
                           _      => Some::<~str>(def.to_owned()) }
}

#[deriving(Eq)]
pub enum FailType {
    ArgumentMissing_,
    UnrecognizedOption_,
    OptionMissing_,
    OptionDuplicated_,
    UnexpectedArgument_,
}

/** A module which provides a way to specify descriptions and
 *  groups of short and long option names, together.
 */
pub mod groups {
    use getopts::{HasArg, Long, Maybe, Multi, No, Occur, Opt, Optional, Req};
    use getopts::{Short, Yes};

    use std::vec;

    /** one group of options, e.g., both -h and --help, along with
     * their shared description and properties
     */
    #[deriving(Clone, Eq)]
    pub struct OptGroup {
        short_name: ~str,
        long_name: ~str,
        hint: ~str,
        desc: ~str,
        hasarg: HasArg,
        occur: Occur
    }

    /// Create a long option that is required and takes an argument
    pub fn reqopt(short_name: &str, long_name: &str,
                  desc: &str, hint: &str) -> OptGroup {
        let len = short_name.len();
        assert!(len == 1 || len == 0);
        return OptGroup { short_name: short_name.to_owned(),
                long_name: long_name.to_owned(),
                hint: hint.to_owned(),
                desc: desc.to_owned(),
                hasarg: Yes,
                occur: Req};
    }

    /// Create a long option that is optional and takes an argument
    pub fn optopt(short_name: &str, long_name: &str,
                  desc: &str, hint: &str) -> OptGroup {
        let len = short_name.len();
        assert!(len == 1 || len == 0);
        return OptGroup {short_name: short_name.to_owned(),
                long_name: long_name.to_owned(),
                hint: hint.to_owned(),
                desc: desc.to_owned(),
                hasarg: Yes,
                occur: Optional};
    }

    /// Create a long option that is optional and does not take an argument
    pub fn optflag(short_name: &str, long_name: &str,
                   desc: &str) -> OptGroup {
        let len = short_name.len();
        assert!(len == 1 || len == 0);
        return OptGroup {short_name: short_name.to_owned(),
                long_name: long_name.to_owned(),
                hint: ~"",
                desc: desc.to_owned(),
                hasarg: No,
                occur: Optional};
    }

    /// Create a long option that is optional and takes an optional argument
    pub fn optflagopt(short_name: &str, long_name: &str,
                      desc: &str, hint: &str) -> OptGroup {
        let len = short_name.len();
        assert!(len == 1 || len == 0);
        return OptGroup {short_name: short_name.to_owned(),
                long_name: long_name.to_owned(),
                hint: hint.to_owned(),
                desc: desc.to_owned(),
                hasarg: Maybe,
                occur: Optional};
    }

    /**
     * Create a long option that is optional, takes an argument, and may occur
     * multiple times
     */
    pub fn optmulti(short_name: &str, long_name: &str,
                    desc: &str, hint: &str) -> OptGroup {
        let len = short_name.len();
        assert!(len == 1 || len == 0);
        return OptGroup {short_name: short_name.to_owned(),
                long_name: long_name.to_owned(),
                hint: hint.to_owned(),
                desc: desc.to_owned(),
                hasarg: Yes,
                occur: Multi};
    }

    // translate OptGroup into Opt
    // (both short and long names correspond to different Opts)
    pub fn long_to_short(lopt: &OptGroup) -> ~[Opt] {
        let OptGroup{short_name: short_name,
                     long_name: long_name,
                     hasarg: hasarg,
                     occur: occur,
                     _} = (*lopt).clone();

        match (short_name.len(), long_name.len()) {
           (0,0) => fail!("this long-format option was given no name"),

           (0,_) => ~[Opt {name: Long((long_name)),
                           hasarg: hasarg,
                           occur: occur}],

           (1,0) => ~[Opt {name: Short(short_name.char_at(0)),
                           hasarg: hasarg,
                           occur: occur}],

           (1,_) => ~[Opt {name: Short(short_name.char_at(0)),
                           hasarg: hasarg,
                           occur:  occur},
                      Opt {name:   Long((long_name)),
                           hasarg: hasarg,
                           occur:  occur}],

           (_,_) => fail!("something is wrong with the long-form opt")
        }
    }

    /*
     * Parse command line args with the provided long format options
     */
    pub fn getopts(args: &[~str], opts: &[OptGroup]) -> ::getopts::Result {
        ::getopts::getopts(args, vec::flat_map(opts, long_to_short))
    }

    /**
     * Derive a usage message from a set of long options
     */
    pub fn usage(brief: &str, opts: &[OptGroup]) -> ~str {

        let desc_sep = "\n" + " ".repeat(24);

        let mut rows = opts.iter().transform(|optref| {
            let OptGroup{short_name: short_name,
                         long_name: long_name,
                         hint: hint,
                         desc: desc,
                         hasarg: hasarg,
                         _} = (*optref).clone();

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

            // FIXME: #5516
            // here we just need to indent the start of the description
            let rowlen = row.len();
            if rowlen < 24 {
                for (24 - rowlen).times {
                    row.push_char(' ')
                }
            } else {
                row.push_str(desc_sep)
            }

            // Normalize desc to contain words separated by one space character
            let mut desc_normalized_whitespace = ~"";
            foreach word in desc.word_iter() {
                desc_normalized_whitespace.push_str(word);
                desc_normalized_whitespace.push_char(' ');
            }

            // FIXME: #5516
            let mut desc_rows = ~[];
            for each_split_within(desc_normalized_whitespace, 54) |substr| {
                desc_rows.push(substr.to_owned());
            }

            // FIXME: #5516
            // wrapped description
            row.push_str(desc_rows.connect(desc_sep));

            row
        });

        return brief.to_owned() +
               "\n\nOptions:\n" +
               rows.collect::<~[~str]>().connect("\n") +
               "\n";
    }

    /** Splits a string into substrings with possibly internal whitespace,
     *  each of them at most `lim` bytes long. The substrings have leading and trailing
     *  whitespace removed, and are only cut at whitespace boundaries.
     *
     *  Note: Function was moved here from `std::str` because this module is the only place that
     *  uses it, and because it was to specific for a general string function.
     *
     *  #Failure:
     *
     *  Fails during iteration if the string contains a non-whitespace
     *  sequence longer than the limit.
     */
    priv fn each_split_within<'a>(ss: &'a str,
                                lim: uint,
                                it: &fn(&'a str) -> bool) -> bool {
        // Just for fun, let's write this as an state machine:

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
        let slice: &fn() = || { cont = it(ss.slice(slice_start, last_end)) };

        // if the limit is larger than the string, lower it to save cycles
        if (lim >= fake_i) {
            lim = fake_i;
        }

        let machine: &fn((uint, char)) -> bool = |(i, c)| {
            let whitespace = if ::std::char::is_whitespace(c) { Ws }       else { Cr };
            let limit      = if (i - slice_start + 1) <= lim  { UnderLim } else { OverLim };

            state = match (state, whitespace, limit) {
                (A, Ws, _)        => { A }
                (A, Cr, _)        => { slice_start = i; last_start = i; B }

                (B, Cr, UnderLim) => { B }
                (B, Cr, OverLim)  if (i - last_start + 1) > lim
                                => fail!("word starting with %? longer than limit!",
                                        ss.slice(last_start, i + 1)),
                (B, Cr, OverLim)  => { slice(); slice_start = last_start; B }
                (B, Ws, UnderLim) => { last_end = i; C }
                (B, Ws, OverLim)  => { last_end = i; slice(); A }

                (C, Cr, UnderLim) => { last_start = i; B }
                (C, Cr, OverLim)  => { slice(); slice_start = i; last_start = i; last_end = i; B }
                (C, Ws, OverLim)  => { slice(); A }
                (C, Ws, UnderLim) => { C }
            };

            cont
        };

        ss.iter().enumerate().advance(|x| machine(x));

        // Let the automaton 'run out' by supplying trailing whitespace
        while cont && match state { B | C => true, A => false } {
            machine((fake_i, ' '));
            fake_i += 1;
        }
        return cont;
    }

    #[test]
    priv fn test_split_within() {
        fn t(s: &str, i: uint, u: &[~str]) {
            let mut v = ~[];
            for each_split_within(s, i) |s| { v.push(s.to_owned()) }
            assert!(v.iter().zip(u.iter()).all(|(a,b)| a == b));
        }
        t("", 0, []);
        t("", 15, []);
        t("hello", 15, [~"hello"]);
        t("\nMary had a little lamb\nLittle lamb\n", 15,
            [~"Mary had a", ~"little lamb", ~"Little lamb"]);
        t("\nMary had a little lamb\nLittle lamb\n", ::std::uint::max_value,
            [~"Mary had a little lamb\nLittle lamb"]);
    }
} // end groups module

#[cfg(test)]
mod tests {

    use getopts::groups::OptGroup;
    use getopts::*;

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
    fn test_reqopt_long() {
        let args = ~[~"--test=20"];
        let opts = ~[reqopt("test")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert!((opt_present(m, "test")));
            assert_eq!(opt_str(m, "test"), ~"20");
          }
          _ => { fail!("test_reqopt_long failed"); }
        }
    }

    #[test]
    fn test_reqopt_long_missing() {
        let args = ~[~"blah"];
        let opts = ~[reqopt("test")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionMissing_),
          _ => fail!()
        }
    }

    #[test]
    fn test_reqopt_long_no_arg() {
        let args = ~[~"--test"];
        let opts = ~[reqopt("test")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
    }

    #[test]
    fn test_reqopt_long_multi() {
        let args = ~[~"--test=20", ~"--test=30"];
        let opts = ~[reqopt("test")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail!()
        }
    }

    #[test]
    fn test_reqopt_short() {
        let args = ~[~"-t", ~"20"];
        let opts = ~[reqopt("t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert!((opt_present(m, "t")));
            assert_eq!(opt_str(m, "t"), ~"20");
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_reqopt_short_missing() {
        let args = ~[~"blah"];
        let opts = ~[reqopt("t")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionMissing_),
          _ => fail!()
        }
    }

    #[test]
    fn test_reqopt_short_no_arg() {
        let args = ~[~"-t"];
        let opts = ~[reqopt("t")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
    }

    #[test]
    fn test_reqopt_short_multi() {
        let args = ~[~"-t", ~"20", ~"-t", ~"30"];
        let opts = ~[reqopt("t")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail!()
        }
    }


    // Tests for optopt
    #[test]
    fn test_optopt_long() {
        let args = ~[~"--test=20"];
        let opts = ~[optopt("test")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert!((opt_present(m, "test")));
            assert_eq!(opt_str(m, "test"), ~"20");
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optopt_long_missing() {
        let args = ~[~"blah"];
        let opts = ~[optopt("test")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => assert!(!opt_present(m, "test")),
          _ => fail!()
        }
    }

    #[test]
    fn test_optopt_long_no_arg() {
        let args = ~[~"--test"];
        let opts = ~[optopt("test")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
    }

    #[test]
    fn test_optopt_long_multi() {
        let args = ~[~"--test=20", ~"--test=30"];
        let opts = ~[optopt("test")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail!()
        }
    }

    #[test]
    fn test_optopt_short() {
        let args = ~[~"-t", ~"20"];
        let opts = ~[optopt("t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert!((opt_present(m, "t")));
            assert_eq!(opt_str(m, "t"), ~"20");
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optopt_short_missing() {
        let args = ~[~"blah"];
        let opts = ~[optopt("t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => assert!(!opt_present(m, "t")),
          _ => fail!()
        }
    }

    #[test]
    fn test_optopt_short_no_arg() {
        let args = ~[~"-t"];
        let opts = ~[optopt("t")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
    }

    #[test]
    fn test_optopt_short_multi() {
        let args = ~[~"-t", ~"20", ~"-t", ~"30"];
        let opts = ~[optopt("t")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail!()
        }
    }


    // Tests for optflag
    #[test]
    fn test_optflag_long() {
        let args = ~[~"--test"];
        let opts = ~[optflag("test")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => assert!(opt_present(m, "test")),
          _ => fail!()
        }
    }

    #[test]
    fn test_optflag_long_missing() {
        let args = ~[~"blah"];
        let opts = ~[optflag("test")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => assert!(!opt_present(m, "test")),
          _ => fail!()
        }
    }

    #[test]
    fn test_optflag_long_arg() {
        let args = ~[~"--test=20"];
        let opts = ~[optflag("test")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => {
            error!(fail_str(f.clone()));
            check_fail_type(f, UnexpectedArgument_);
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflag_long_multi() {
        let args = ~[~"--test", ~"--test"];
        let opts = ~[optflag("test")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail!()
        }
    }

    #[test]
    fn test_optflag_short() {
        let args = ~[~"-t"];
        let opts = ~[optflag("t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => assert!(opt_present(m, "t")),
          _ => fail!()
        }
    }

    #[test]
    fn test_optflag_short_missing() {
        let args = ~[~"blah"];
        let opts = ~[optflag("t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => assert!(!opt_present(m, "t")),
          _ => fail!()
        }
    }

    #[test]
    fn test_optflag_short_arg() {
        let args = ~[~"-t", ~"20"];
        let opts = ~[optflag("t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            // The next variable after the flag is just a free argument

            assert!(m.free[0] == ~"20");
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflag_short_multi() {
        let args = ~[~"-t", ~"-t"];
        let opts = ~[optflag("t")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail!()
        }
    }

    // Tests for optflagmulti
    #[test]
    fn test_optflagmulti_short1() {
        let args = ~[~"-v"];
        let opts = ~[optflagmulti("v")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert_eq!(opt_count(m, "v"), 1);
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflagmulti_short2a() {
        let args = ~[~"-v", ~"-v"];
        let opts = ~[optflagmulti("v")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert_eq!(opt_count(m, "v"), 2);
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflagmulti_short2b() {
        let args = ~[~"-vv"];
        let opts = ~[optflagmulti("v")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert_eq!(opt_count(m, "v"), 2);
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflagmulti_long1() {
        let args = ~[~"--verbose"];
        let opts = ~[optflagmulti("verbose")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert_eq!(opt_count(m, "verbose"), 1);
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optflagmulti_long2() {
        let args = ~[~"--verbose", ~"--verbose"];
        let opts = ~[optflagmulti("verbose")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert_eq!(opt_count(m, "verbose"), 2);
          }
          _ => fail!()
        }
    }

    // Tests for optmulti
    #[test]
    fn test_optmulti_long() {
        let args = ~[~"--test=20"];
        let opts = ~[optmulti("test")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert!((opt_present(m, "test")));
            assert_eq!(opt_str(m, "test"), ~"20");
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optmulti_long_missing() {
        let args = ~[~"blah"];
        let opts = ~[optmulti("test")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => assert!(!opt_present(m, "test")),
          _ => fail!()
        }
    }

    #[test]
    fn test_optmulti_long_no_arg() {
        let args = ~[~"--test"];
        let opts = ~[optmulti("test")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
    }

    #[test]
    fn test_optmulti_long_multi() {
        let args = ~[~"--test=20", ~"--test=30"];
        let opts = ~[optmulti("test")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
              assert!(opt_present(m, "test"));
              assert_eq!(opt_str(m, "test"), ~"20");
              let pair = opt_strs(m, "test");
              assert!(pair[0] == ~"20");
              assert!(pair[1] == ~"30");
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optmulti_short() {
        let args = ~[~"-t", ~"20"];
        let opts = ~[optmulti("t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert!((opt_present(m, "t")));
            assert_eq!(opt_str(m, "t"), ~"20");
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_optmulti_short_missing() {
        let args = ~[~"blah"];
        let opts = ~[optmulti("t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => assert!(!opt_present(m, "t")),
          _ => fail!()
        }
    }

    #[test]
    fn test_optmulti_short_no_arg() {
        let args = ~[~"-t"];
        let opts = ~[optmulti("t")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail!()
        }
    }

    #[test]
    fn test_optmulti_short_multi() {
        let args = ~[~"-t", ~"20", ~"-t", ~"30"];
        let opts = ~[optmulti("t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert!((opt_present(m, "t")));
            assert_eq!(opt_str(m, "t"), ~"20");
            let pair = opt_strs(m, "t");
            assert!(pair[0] == ~"20");
            assert!(pair[1] == ~"30");
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_unrecognized_option_long() {
        let args = ~[~"--untest"];
        let opts = ~[optmulti("t")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, UnrecognizedOption_),
          _ => fail!()
        }
    }

    #[test]
    fn test_unrecognized_option_short() {
        let args = ~[~"-t"];
        let opts = ~[optmulti("test")];
        let rs = getopts(args, opts);
        match rs {
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
            ~[optopt("s"), optflag("flag"), reqopt("long"),
             optflag("f"), optmulti("m"), optmulti("n"),
             optopt("notpresent")];
        let rs = getopts(args, opts);
        match rs {
          Ok(ref m) => {
            assert!(m.free[0] == ~"prog");
            assert!(m.free[1] == ~"free1");
            assert_eq!(opt_str(m, "s"), ~"20");
            assert!(m.free[2] == ~"free2");
            assert!((opt_present(m, "flag")));
            assert_eq!(opt_str(m, "long"), ~"30");
            assert!((opt_present(m, "f")));
            let pair = opt_strs(m, "m");
            assert!(pair[0] == ~"40");
            assert!(pair[1] == ~"50");
            let pair = opt_strs(m, "n");
            assert!(pair[0] == ~"-A B");
            assert!(pair[1] == ~"-60 70");
            assert!((!opt_present(m, "notpresent")));
          }
          _ => fail!()
        }
    }

    #[test]
    fn test_multi() {
        let opts = ~[optopt("e"), optopt("encrypt"), optopt("f")];

        let args_single = ~[~"-e", ~"foo"];
        let matches_single = &match getopts(args_single, opts) {
          result::Ok(m) => m,
          result::Err(_) => fail!()
        };
        assert!(opts_present(matches_single, [~"e"]));
        assert!(opts_present(matches_single, [~"encrypt", ~"e"]));
        assert!(opts_present(matches_single, [~"e", ~"encrypt"]));
        assert!(!opts_present(matches_single, [~"encrypt"]));
        assert!(!opts_present(matches_single, [~"thing"]));
        assert!(!opts_present(matches_single, []));

        assert_eq!(opts_str(matches_single, [~"e"]), ~"foo");
        assert_eq!(opts_str(matches_single, [~"e", ~"encrypt"]), ~"foo");
        assert_eq!(opts_str(matches_single, [~"encrypt", ~"e"]), ~"foo");

        let args_both = ~[~"-e", ~"foo", ~"--encrypt", ~"foo"];
        let matches_both = &match getopts(args_both, opts) {
          result::Ok(m) => m,
          result::Err(_) => fail!()
        };
        assert!(opts_present(matches_both, [~"e"]));
        assert!(opts_present(matches_both, [~"encrypt"]));
        assert!(opts_present(matches_both, [~"encrypt", ~"e"]));
        assert!(opts_present(matches_both, [~"e", ~"encrypt"]));
        assert!(!opts_present(matches_both, [~"f"]));
        assert!(!opts_present(matches_both, [~"thing"]));
        assert!(!opts_present(matches_both, []));

        assert_eq!(opts_str(matches_both, [~"e"]), ~"foo");
        assert_eq!(opts_str(matches_both, [~"encrypt"]), ~"foo");
        assert_eq!(opts_str(matches_both, [~"e", ~"encrypt"]), ~"foo");
        assert_eq!(opts_str(matches_both, [~"encrypt", ~"e"]), ~"foo");
    }

    #[test]
    fn test_nospace() {
        let args = ~[~"-Lfoo", ~"-M."];
        let opts = ~[optmulti("L"), optmulti("M")];
        let matches = &match getopts(args, opts) {
          result::Ok(m) => m,
          result::Err(_) => fail!()
        };
        assert!(opts_present(matches, [~"L"]));
        assert_eq!(opts_str(matches, [~"L"]), ~"foo");
        assert!(opts_present(matches, [~"M"]));
        assert_eq!(opts_str(matches, [~"M"]), ~".");

    }

    #[test]
    fn test_groups_reqopt() {
        let opt = groups::reqopt("b", "banana", "some bananas", "VAL");
        assert!(opt == OptGroup { short_name: ~"b",
                        long_name: ~"banana",
                        hint: ~"VAL",
                        desc: ~"some bananas",
                        hasarg: Yes,
                        occur: Req })
    }

    #[test]
    fn test_groups_optopt() {
        let opt = groups::optopt("a", "apple", "some apples", "VAL");
        assert!(opt == OptGroup { short_name: ~"a",
                        long_name: ~"apple",
                        hint: ~"VAL",
                        desc: ~"some apples",
                        hasarg: Yes,
                        occur: Optional })
    }

    #[test]
    fn test_groups_optflag() {
        let opt = groups::optflag("k", "kiwi", "some kiwis");
        assert!(opt == OptGroup { short_name: ~"k",
                        long_name: ~"kiwi",
                        hint: ~"",
                        desc: ~"some kiwis",
                        hasarg: No,
                        occur: Optional })
    }

    #[test]
    fn test_groups_optflagopt() {
        let opt = groups::optflagopt("p", "pineapple", "some pineapples", "VAL");
        assert!(opt == OptGroup { short_name: ~"p",
                        long_name: ~"pineapple",
                        hint: ~"VAL",
                        desc: ~"some pineapples",
                        hasarg: Maybe,
                        occur: Optional })
    }

    #[test]
    fn test_groups_optmulti() {
        let opt = groups::optmulti("l", "lime", "some limes", "VAL");
        assert!(opt == OptGroup { short_name: ~"l",
                        long_name: ~"lime",
                        hint: ~"VAL",
                        desc: ~"some limes",
                        hasarg: Yes,
                        occur: Multi })
    }

    #[test]
    fn test_groups_long_to_short() {
        let short = ~[reqopt("b"), reqopt("banana")];
        let verbose = groups::reqopt("b", "banana", "some bananas", "VAL");

        assert_eq!(groups::long_to_short(&verbose), short);
    }

    #[test]
    fn test_groups_getopts() {
        let short = ~[
            reqopt("b"), reqopt("banana"),
            optopt("a"), optopt("apple"),
            optflag("k"), optflagopt("kiwi"),
            optflagopt("p"),
            optmulti("l")
        ];

        let verbose = ~[
            groups::reqopt("b", "banana", "Desc", "VAL"),
            groups::optopt("a", "apple", "Desc", "VAL"),
            groups::optflag("k", "kiwi", "Desc"),
            groups::optflagopt("p", "", "Desc", "VAL"),
            groups::optmulti("l", "", "Desc", "VAL"),
        ];

        let sample_args = ~[~"-k", ~"15", ~"--apple", ~"1", ~"k",
                            ~"-p", ~"16", ~"l", ~"35"];

        // FIXME #4681: sort options here?
        assert!(getopts(sample_args, short)
            == groups::getopts(sample_args, verbose));
    }

    #[test]
    fn test_groups_usage() {
        let optgroups = ~[
            groups::reqopt("b", "banana", "Desc", "VAL"),
            groups::optopt("a", "012345678901234567890123456789",
                             "Desc", "VAL"),
            groups::optflag("k", "kiwi", "Desc"),
            groups::optflagopt("p", "", "Desc", "VAL"),
            groups::optmulti("l", "", "Desc", "VAL"),
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

        let generated_usage = groups::usage("Usage: fruits", optgroups);

        debug!("expected: <<%s>>", expected);
        debug!("generated: <<%s>>", generated_usage);
        assert_eq!(generated_usage, expected);
    }

    #[test]
    fn test_groups_usage_description_wrapping() {
        // indentation should be 24 spaces
        // lines wrap after 78: or rather descriptions wrap after 54

        let optgroups = ~[
            groups::optflag("k", "kiwi",
                "This is a long description which won't be wrapped..+.."), // 54
            groups::optflag("a", "apple",
                "This is a long description which _will_ be wrapped..+.."), // 55
        ];

        let expected =
~"Usage: fruits

Options:
    -k --kiwi           This is a long description which won't be wrapped..+..
    -a --apple          This is a long description which _will_ be
                        wrapped..+..
";

        let usage = groups::usage("Usage: fruits", optgroups);

        debug!("expected: <<%s>>", expected);
        debug!("generated: <<%s>>", usage);
        assert!(usage == expected)
    }
}
