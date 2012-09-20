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
 * argument following either a space or an equals sign.
 *
 * # Example
 *
 * The following example shows simple command line parsing for an application
 * that requires an input file to be specified, accepts an optional output
 * file name following -o, and accepts both -h and --help as optional flags.
 *
 *     use std;
 *     import std::getopts::{optopt,optflag,getopts,opt_present,opt_maybe_str,
 *         fail_str};
 *
 *     fn do_work(in: str, out: Option<str>) {
 *         // ...
 *     }
 *
 *     fn print_usage(program: str) {
 *         io::println("Usage: " + program + " [options]");
 *         io::println("-o\t\tOutput");
 *         io::println("-h --help\tUsage");
 *     }
 *
 *     fn main(args: ~[str]) {
 *         check vec::is_not_empty(args);
 *
 *         let program : str = vec::head(args);
 *
 *         let opts = ~[
 *             optopt("o"),
 *             optflag("h"),
 *             optflag("help")
 *         ];
 *         let matches = match getopts(vec::tail(args), opts) {
 *             result::ok(m) { m }
 *             result::err(f) { fail fail_str(f) }
 *         };
 *         if opt_present(matches, "h") || opt_present(matches, "help") {
 *             print_usage(program);
 *             return;
 *         }
 *         let output = opt_maybe_str(matches, "o");
 *         let input = if vec::is_not_empty(matches.free) {
 *             matches.free[0]
 *         } else {
 *             print_usage(program);
 *             return;
 *         };
 *         do_work(input, output);
 *     }
 */

#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use core::cmp::Eq;
use core::result::{Err, Ok};
use core::option;
use core::option::{Some, None};
export Opt;
export reqopt;
export optopt;
export optflag;
export optflagopt;
export optmulti;
export getopts;
export Matches;
export Fail_;
export fail_str;
export opt_present;
export opts_present;
export opt_str;
export opts_str;
export opt_strs;
export opt_maybe_str;
export opt_default;
export Result; //NDM

enum Name {
    Long(~str),
    Short(char),
}

enum HasArg { Yes, No, Maybe, }

enum Occur { Req, Optional, Multi, }

/// A description of a possible option
type Opt = {name: Name, hasarg: HasArg, occur: Occur};

fn mkname(nm: &str) -> Name {
    let unm = str::from_slice(nm);
    return if str::len(nm) == 1u {
            Short(str::char_at(unm, 0u))
        } else { Long(unm) };
}

#[cfg(stage0)]
impl Name : Eq {
    pure fn eq(&&other: Name) -> bool {
        match self {
            Long(e0a) => {
                match other {
                    Long(e0b) => e0a == e0b,
                    _ => false
                }
            }
            Short(e0a) => {
                match other {
                    Short(e0b) => e0a == e0b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(&&other: Name) -> bool { !self.eq(other) }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl Name : Eq {
    pure fn eq(other: &Name) -> bool {
        match self {
            Long(e0a) => {
                match (*other) {
                    Long(e0b) => e0a == e0b,
                    _ => false
                }
            }
            Short(e0a) => {
                match (*other) {
                    Short(e0b) => e0a == e0b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(other: &Name) -> bool { !self.eq(other) }
}

#[cfg(stage0)]
impl Occur : Eq {
    pure fn eq(&&other: Occur) -> bool {
        (self as uint) == (other as uint)
    }
    pure fn ne(&&other: Occur) -> bool { !self.eq(other) }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl Occur : Eq {
    pure fn eq(other: &Occur) -> bool {
        (self as uint) == ((*other) as uint)
    }
    pure fn ne(other: &Occur) -> bool { !self.eq(other) }
}

/// Create an option that is required and takes an argument
fn reqopt(name: &str) -> Opt {
    return {name: mkname(name), hasarg: Yes, occur: Req};
}

/// Create an option that is optional and takes an argument
fn optopt(name: &str) -> Opt {
    return {name: mkname(name), hasarg: Yes, occur: Optional};
}

/// Create an option that is optional and does not take an argument
fn optflag(name: &str) -> Opt {
    return {name: mkname(name), hasarg: No, occur: Optional};
}

/// Create an option that is optional and takes an optional argument
fn optflagopt(name: &str) -> Opt {
    return {name: mkname(name), hasarg: Maybe, occur: Optional};
}

/**
 * Create an option that is optional, takes an argument, and may occur
 * multiple times
 */
fn optmulti(name: &str) -> Opt {
    return {name: mkname(name), hasarg: Yes, occur: Multi};
}

enum Optval { Val(~str), Given, }

/**
 * The result of checking command line arguments. Contains a vector
 * of matches and a vector of free strings.
 */
type Matches = {opts: ~[Opt], vals: ~[~[Optval]], free: ~[~str]};

fn is_arg(arg: &str) -> bool {
    return str::len(arg) > 1u && arg[0] == '-' as u8;
}

fn name_str(nm: &Name) -> ~str {
    return match *nm {
      Short(ch) => str::from_char(ch),
      Long(s) => s
    };
}

fn find_opt(opts: &[Opt], +nm: Name) -> Option<uint> {
    vec::position(opts, |opt| opt.name == nm)
}

/**
 * The type returned when the command line does not conform to the
 * expected format. Pass this value to <fail_str> to get an error message.
 */
enum Fail_ {
    ArgumentMissing(~str),
    UnrecognizedOption(~str),
    OptionMissing(~str),
    OptionDuplicated(~str),
    UnexpectedArgument(~str),
}

/// Convert a `fail_` enum into an error string
fn fail_str(+f: Fail_) -> ~str {
    return match f {
      ArgumentMissing(nm) => ~"Argument to option '" + nm + ~"' missing.",
      UnrecognizedOption(nm) => ~"Unrecognized option: '" + nm + ~"'.",
      OptionMissing(nm) => ~"Required option '" + nm + ~"' missing.",
      OptionDuplicated(nm) => ~"Option '" + nm + ~"' given more than once.",
      UnexpectedArgument(nm) => {
        ~"Option " + nm + ~" does not take an argument."
      }
    };
}

/**
 * The result of parsing a command line with a set of options
 * (result::t<Matches, Fail_>)
 */
type Result = result::Result<Matches, Fail_>;

/**
 * Parse command line arguments according to the provided options
 *
 * On success returns `ok(Opt)`. Use functions such as `opt_present`
 * `opt_str`, etc. to interrogate results.  Returns `err(Fail_)` on failure.
 * Use <fail_str> to get an error message.
 */
fn getopts(args: &[~str], opts: &[Opt]) -> Result unsafe {
    let n_opts = vec::len::<Opt>(opts);
    fn f(_x: uint) -> ~[Optval] { return ~[]; }
    let vals = vec::to_mut(vec::from_fn(n_opts, f));
    let mut free: ~[~str] = ~[];
    let l = vec::len(args);
    let mut i = 0u;
    while i < l {
        let cur = args[i];
        let curlen = str::len(cur);
        if !is_arg(cur) {
            vec::push(free, cur);
        } else if cur == ~"--" {
            let mut j = i + 1u;
            while j < l { vec::push(free, args[j]); j += 1u; }
            break;
        } else {
            let mut names;
            let mut i_arg = option::None::<~str>;
            if cur[1] == '-' as u8 {
                let tail = str::slice(cur, 2u, curlen);
                let tail_eq = str::splitn_char(tail, '=', 1u);
                if vec::len(tail_eq) <= 1u {
                    names = ~[Long(tail)];
                } else {
                    names =
                        ~[Long(tail_eq[0])];
                    i_arg =
                        option::Some::<~str>(tail_eq[1]);
                }
            } else {
                let mut j = 1u;
                let mut last_valid_opt_id = option::None;
                names = ~[];
                while j < curlen {
                    let range = str::char_range_at(cur, j);
                    let opt = Short(range.ch);

                    /* In a series of potential options (eg. -aheJ), if we see
                       one which takes an argument, we assume all subsequent
                       characters make up the argument. This allows options
                       such as -L/usr/local/lib/foo to be interpreted
                       correctly
                    */

                    match find_opt(opts, opt) {
                      Some(id) => last_valid_opt_id = option::Some(id),
                      None => {
                        let arg_follows =
                            option::is_some(last_valid_opt_id) &&
                            match opts[option::get(last_valid_opt_id)]
                              .hasarg {

                              Yes | Maybe => true,
                              No => false
                            };
                        if arg_follows && j + 1 < curlen {
                            i_arg = option::Some(str::slice(cur, j, curlen));
                            break;
                        } else {
                            last_valid_opt_id = option::None;
                        }
                      }
                    }
                    vec::push(names, opt);
                    j = range.next;
                }
            }
            let mut name_pos = 0u;
            for vec::each(names) |nm| {
                name_pos += 1u;
                let optid = match find_opt(opts, *nm) {
                  Some(id) => id,
                  None => return Err(UnrecognizedOption(name_str(nm)))
                };
                match opts[optid].hasarg {
                  No => {
                    if !option::is_none::<~str>(i_arg) {
                        return Err(UnexpectedArgument(name_str(nm)));
                    }
                    vec::push(vals[optid], Given);
                  }
                  Maybe => {
                    if !option::is_none::<~str>(i_arg) {
                        vec::push(vals[optid], Val(option::get(i_arg)));
                    } else if name_pos < vec::len::<Name>(names) ||
                                  i + 1u == l || is_arg(args[i + 1u]) {
                        vec::push(vals[optid], Given);
                    } else { i += 1u; vec::push(vals[optid], Val(args[i])); }
                  }
                  Yes => {
                    if !option::is_none::<~str>(i_arg) {
                        vec::push(vals[optid],
                                  Val(option::get::<~str>(i_arg)));
                    } else if i + 1u == l {
                        return Err(ArgumentMissing(name_str(nm)));
                    } else { i += 1u; vec::push(vals[optid], Val(args[i])); }
                  }
                }
            }
        }
        i += 1u;
    }
    i = 0u;
    while i < n_opts {
        let n = vec::len::<Optval>(vals[i]);
        let occ = opts[i].occur;
        if occ == Req {
            if n == 0u {
                return Err(OptionMissing(name_str(&(opts[i].name))));
            }
        }
        if occ != Multi {
            if n > 1u {
                return Err(OptionDuplicated(name_str(&(opts[i].name))));
            }
        }
        i += 1u;
    }
    return Ok({opts: vec::from_slice(opts),
               vals: vec::from_mut(move vals),
               free: free});
}

fn opt_vals(+mm: Matches, nm: &str) -> ~[Optval] {
    return match find_opt(mm.opts, mkname(nm)) {
      Some(id) => mm.vals[id],
      None => {
        error!("No option '%s' defined", nm);
        fail
      }
    };
}

fn opt_val(+mm: Matches, nm: &str) -> Optval { return opt_vals(mm, nm)[0]; }

/// Returns true if an option was matched
fn opt_present(+mm: Matches, nm: &str) -> bool {
    return vec::len::<Optval>(opt_vals(mm, nm)) > 0u;
}

/// Returns true if any of several options were matched
fn opts_present(+mm: Matches, names: &[~str]) -> bool {
    for vec::each(names) |nm| {
        match find_opt(mm.opts, mkname(*nm)) {
          Some(_) => return true,
          None    => ()
        }
    }
    return false;
}


/**
 * Returns the string argument supplied to a matching option
 *
 * Fails if the option was not matched or if the match did not take an
 * argument
 */
fn opt_str(+mm: Matches, nm: &str) -> ~str {
    return match opt_val(mm, nm) { Val(s) => s, _ => fail };
}

/**
 * Returns the string argument supplied to one of several matching options
 *
 * Fails if the no option was provided from the given list, or if the no such
 * option took an argument
 */
fn opts_str(+mm: Matches, names: &[~str]) -> ~str {
    for vec::each(names) |nm| {
        match opt_val(mm, *nm) {
          Val(s) => return s,
          _ => ()
        }
    }
    fail;
}


/**
 * Returns a vector of the arguments provided to all matches of the given
 * option.
 *
 * Used when an option accepts multiple values.
 */
fn opt_strs(+mm: Matches, nm: &str) -> ~[~str] {
    let mut acc: ~[~str] = ~[];
    for vec::each(opt_vals(mm, nm)) |v| {
        match *v { Val(s) => vec::push(acc, s), _ => () }
    }
    return acc;
}

/// Returns the string argument supplied to a matching option or none
fn opt_maybe_str(+mm: Matches, nm: &str) -> Option<~str> {
    let vals = opt_vals(mm, nm);
    if vec::len::<Optval>(vals) == 0u { return None::<~str>; }
    return match vals[0] { Val(s) => Some::<~str>(s), _ => None::<~str> };
}


/**
 * Returns the matching string, a default, or none
 *
 * Returns none if the option was not present, `def` if the option was
 * present but no argument was provided, and the argument if the option was
 * present and an argument was provided.
 */
fn opt_default(+mm: Matches, nm: &str, def: &str) -> Option<~str> {
    let vals = opt_vals(mm, nm);
    if vec::len::<Optval>(vals) == 0u { return None::<~str>; }
    return match vals[0] { Val(s) => Some::<~str>(s),
                           _      => Some::<~str>(str::from_slice(def)) }
}

enum FailType {
    ArgumentMissing_,
    UnrecognizedOption_,
    OptionMissing_,
    OptionDuplicated_,
    UnexpectedArgument_,
}

#[cfg(stage0)]
impl FailType : Eq {
    pure fn eq(&&other: FailType) -> bool {
        (self as uint) == (other as uint)
    }
    pure fn ne(&&other: FailType) -> bool { !self.eq(other) }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl FailType : Eq {
    pure fn eq(other: &FailType) -> bool {
        (self as uint) == ((*other) as uint)
    }
    pure fn ne(other: &FailType) -> bool { !self.eq(other) }
}

#[cfg(test)]
mod tests {
    use opt = getopts;
    use result::{Err, Ok};

    fn check_fail_type(+f: Fail_, ft: FailType) {
        match f {
          ArgumentMissing(_) => assert ft == ArgumentMissing_,
          UnrecognizedOption(_) => assert ft == UnrecognizedOption_,
          OptionMissing(_) => assert ft == OptionMissing_,
          OptionDuplicated(_) => assert ft == OptionDuplicated_,
          UnexpectedArgument(_) => assert ft == UnexpectedArgument_
        }
    }


    // Tests for reqopt
    #[test]
    fn test_reqopt_long() {
        let args = ~[~"--test=20"];
        let opts = ~[reqopt(~"test")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => {
            assert (opt_present(m, ~"test"));
            assert (opt_str(m, ~"test") == ~"20");
          }
          _ => { fail ~"test_reqopt_long failed"; }
        }
    }

    #[test]
    fn test_reqopt_long_missing() {
        let args = ~[~"blah"];
        let opts = ~[reqopt(~"test")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionMissing_),
          _ => fail
        }
    }

    #[test]
    fn test_reqopt_long_no_arg() {
        let args = ~[~"--test"];
        let opts = ~[reqopt(~"test")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail
        }
    }

    #[test]
    fn test_reqopt_long_multi() {
        let args = ~[~"--test=20", ~"--test=30"];
        let opts = ~[reqopt(~"test")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail
        }
    }

    #[test]
    fn test_reqopt_short() {
        let args = ~[~"-t", ~"20"];
        let opts = ~[reqopt(~"t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => {
            assert (opt_present(m, ~"t"));
            assert (opt_str(m, ~"t") == ~"20");
          }
          _ => fail
        }
    }

    #[test]
    fn test_reqopt_short_missing() {
        let args = ~[~"blah"];
        let opts = ~[reqopt(~"t")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionMissing_),
          _ => fail
        }
    }

    #[test]
    fn test_reqopt_short_no_arg() {
        let args = ~[~"-t"];
        let opts = ~[reqopt(~"t")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail
        }
    }

    #[test]
    fn test_reqopt_short_multi() {
        let args = ~[~"-t", ~"20", ~"-t", ~"30"];
        let opts = ~[reqopt(~"t")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail
        }
    }


    // Tests for optopt
    #[test]
    fn test_optopt_long() {
        let args = ~[~"--test=20"];
        let opts = ~[optopt(~"test")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => {
            assert (opt_present(m, ~"test"));
            assert (opt_str(m, ~"test") == ~"20");
          }
          _ => fail
        }
    }

    #[test]
    fn test_optopt_long_missing() {
        let args = ~[~"blah"];
        let opts = ~[optopt(~"test")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => assert (!opt_present(m, ~"test")),
          _ => fail
        }
    }

    #[test]
    fn test_optopt_long_no_arg() {
        let args = ~[~"--test"];
        let opts = ~[optopt(~"test")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail
        }
    }

    #[test]
    fn test_optopt_long_multi() {
        let args = ~[~"--test=20", ~"--test=30"];
        let opts = ~[optopt(~"test")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail
        }
    }

    #[test]
    fn test_optopt_short() {
        let args = ~[~"-t", ~"20"];
        let opts = ~[optopt(~"t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => {
            assert (opt_present(m, ~"t"));
            assert (opt_str(m, ~"t") == ~"20");
          }
          _ => fail
        }
    }

    #[test]
    fn test_optopt_short_missing() {
        let args = ~[~"blah"];
        let opts = ~[optopt(~"t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => assert (!opt_present(m, ~"t")),
          _ => fail
        }
    }

    #[test]
    fn test_optopt_short_no_arg() {
        let args = ~[~"-t"];
        let opts = ~[optopt(~"t")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail
        }
    }

    #[test]
    fn test_optopt_short_multi() {
        let args = ~[~"-t", ~"20", ~"-t", ~"30"];
        let opts = ~[optopt(~"t")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail
        }
    }


    // Tests for optflag
    #[test]
    fn test_optflag_long() {
        let args = ~[~"--test"];
        let opts = ~[optflag(~"test")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => assert (opt_present(m, ~"test")),
          _ => fail
        }
    }

    #[test]
    fn test_optflag_long_missing() {
        let args = ~[~"blah"];
        let opts = ~[optflag(~"test")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => assert (!opt_present(m, ~"test")),
          _ => fail
        }
    }

    #[test]
    fn test_optflag_long_arg() {
        let args = ~[~"--test=20"];
        let opts = ~[optflag(~"test")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => {
            log(error, fail_str(f));
            check_fail_type(f, UnexpectedArgument_);
          }
          _ => fail
        }
    }

    #[test]
    fn test_optflag_long_multi() {
        let args = ~[~"--test", ~"--test"];
        let opts = ~[optflag(~"test")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail
        }
    }

    #[test]
    fn test_optflag_short() {
        let args = ~[~"-t"];
        let opts = ~[optflag(~"t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => assert (opt_present(m, ~"t")),
          _ => fail
        }
    }

    #[test]
    fn test_optflag_short_missing() {
        let args = ~[~"blah"];
        let opts = ~[optflag(~"t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => assert (!opt_present(m, ~"t")),
          _ => fail
        }
    }

    #[test]
    fn test_optflag_short_arg() {
        let args = ~[~"-t", ~"20"];
        let opts = ~[optflag(~"t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => {
            // The next variable after the flag is just a free argument

            assert (m.free[0] == ~"20");
          }
          _ => fail
        }
    }

    #[test]
    fn test_optflag_short_multi() {
        let args = ~[~"-t", ~"-t"];
        let opts = ~[optflag(~"t")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, OptionDuplicated_),
          _ => fail
        }
    }


    // Tests for optmulti
    #[test]
    fn test_optmulti_long() {
        let args = ~[~"--test=20"];
        let opts = ~[optmulti(~"test")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => {
            assert (opt_present(m, ~"test"));
            assert (opt_str(m, ~"test") == ~"20");
          }
          _ => fail
        }
    }

    #[test]
    fn test_optmulti_long_missing() {
        let args = ~[~"blah"];
        let opts = ~[optmulti(~"test")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => assert (!opt_present(m, ~"test")),
          _ => fail
        }
    }

    #[test]
    fn test_optmulti_long_no_arg() {
        let args = ~[~"--test"];
        let opts = ~[optmulti(~"test")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail
        }
    }

    #[test]
    fn test_optmulti_long_multi() {
        let args = ~[~"--test=20", ~"--test=30"];
        let opts = ~[optmulti(~"test")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => {
              assert (opt_present(m, ~"test"));
              assert (opt_str(m, ~"test") == ~"20");
              let pair = opt_strs(m, ~"test");
              assert (pair[0] == ~"20");
              assert (pair[1] == ~"30");
          }
          _ => fail
        }
    }

    #[test]
    fn test_optmulti_short() {
        let args = ~[~"-t", ~"20"];
        let opts = ~[optmulti(~"t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => {
            assert (opt_present(m, ~"t"));
            assert (opt_str(m, ~"t") == ~"20");
          }
          _ => fail
        }
    }

    #[test]
    fn test_optmulti_short_missing() {
        let args = ~[~"blah"];
        let opts = ~[optmulti(~"t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => assert (!opt_present(m, ~"t")),
          _ => fail
        }
    }

    #[test]
    fn test_optmulti_short_no_arg() {
        let args = ~[~"-t"];
        let opts = ~[optmulti(~"t")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, ArgumentMissing_),
          _ => fail
        }
    }

    #[test]
    fn test_optmulti_short_multi() {
        let args = ~[~"-t", ~"20", ~"-t", ~"30"];
        let opts = ~[optmulti(~"t")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => {
            assert (opt_present(m, ~"t"));
            assert (opt_str(m, ~"t") == ~"20");
            let pair = opt_strs(m, ~"t");
            assert (pair[0] == ~"20");
            assert (pair[1] == ~"30");
          }
          _ => fail
        }
    }

    #[test]
    fn test_unrecognized_option_long() {
        let args = ~[~"--untest"];
        let opts = ~[optmulti(~"t")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, UnrecognizedOption_),
          _ => fail
        }
    }

    #[test]
    fn test_unrecognized_option_short() {
        let args = ~[~"-t"];
        let opts = ~[optmulti(~"test")];
        let rs = getopts(args, opts);
        match rs {
          Err(f) => check_fail_type(f, UnrecognizedOption_),
          _ => fail
        }
    }

    #[test]
    fn test_combined() {
        let args =
            ~[~"prog", ~"free1", ~"-s", ~"20", ~"free2",
              ~"--flag", ~"--long=30", ~"-f", ~"-m", ~"40",
              ~"-m", ~"50", ~"-n", ~"-A B", ~"-n", ~"-60 70"];
        let opts =
            ~[optopt(~"s"), optflag(~"flag"), reqopt(~"long"),
             optflag(~"f"), optmulti(~"m"), optmulti(~"n"),
             optopt(~"notpresent")];
        let rs = getopts(args, opts);
        match rs {
          Ok(m) => {
            assert (m.free[0] == ~"prog");
            assert (m.free[1] == ~"free1");
            assert (opt_str(m, ~"s") == ~"20");
            assert (m.free[2] == ~"free2");
            assert (opt_present(m, ~"flag"));
            assert (opt_str(m, ~"long") == ~"30");
            assert (opt_present(m, ~"f"));
            let pair = opt_strs(m, ~"m");
            assert (pair[0] == ~"40");
            assert (pair[1] == ~"50");
            let pair = opt_strs(m, ~"n");
            assert (pair[0] == ~"-A B");
            assert (pair[1] == ~"-60 70");
            assert (!opt_present(m, ~"notpresent"));
          }
          _ => fail
        }
    }

    #[test]
    fn test_multi() {
        let args = ~[~"-e", ~"foo", ~"--encrypt", ~"foo"];
        let opts = ~[optopt(~"e"), optopt(~"encrypt")];
        let matches = match getopts(args, opts) {
          result::Ok(m) => m,
          result::Err(_f) => fail
        };
        assert opts_present(matches, ~[~"e"]);
        assert opts_present(matches, ~[~"encrypt"]);
        assert opts_present(matches, ~[~"encrypt", ~"e"]);
        assert opts_present(matches, ~[~"e", ~"encrypt"]);
        assert !opts_present(matches, ~[~"thing"]);
        assert !opts_present(matches, ~[]);

        assert opts_str(matches, ~[~"e"]) == ~"foo";
        assert opts_str(matches, ~[~"encrypt"]) == ~"foo";
        assert opts_str(matches, ~[~"e", ~"encrypt"]) == ~"foo";
        assert opts_str(matches, ~[~"encrypt", ~"e"]) == ~"foo";
    }

    #[test]
    fn test_nospace() {
        let args = ~[~"-Lfoo"];
        let opts = ~[optmulti(~"L")];
        let matches = match getopts(args, opts) {
          result::Ok(m) => m,
          result::Err(_f) => fail
        };
        assert opts_present(matches, ~[~"L"]);
        assert opts_str(matches, ~[~"L"]) == ~"foo";
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
