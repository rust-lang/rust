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

import core::result::{err, ok};
import core::option;
import core::option::{Some, None};
export opt;
export reqopt;
export optopt;
export optflag;
export optflagopt;
export optmulti;
export getopts;
export matches;
export fail_;
export fail_str;
export opt_present;
export opts_present;
export opt_str;
export opts_str;
export opt_strs;
export opt_maybe_str;
export opt_default;
export result; //NDM

enum name { long(~str), short(char), }

enum hasarg { yes, no, maybe, }

enum occur { req, optional, multi, }

/// A description of a possible option
type opt = {name: name, hasarg: hasarg, occur: occur};

fn mkname(nm: ~str) -> name {
    return if str::len(nm) == 1u {
            short(str::char_at(nm, 0u))
        } else { long(nm) };
}

/// Create an option that is required and takes an argument
fn reqopt(name: ~str) -> opt {
    return {name: mkname(name), hasarg: yes, occur: req};
}

/// Create an option that is optional and takes an argument
fn optopt(name: ~str) -> opt {
    return {name: mkname(name), hasarg: yes, occur: optional};
}

/// Create an option that is optional and does not take an argument
fn optflag(name: ~str) -> opt {
    return {name: mkname(name), hasarg: no, occur: optional};
}

/// Create an option that is optional and takes an optional argument
fn optflagopt(name: ~str) -> opt {
    return {name: mkname(name), hasarg: maybe, occur: optional};
}

/**
 * Create an option that is optional, takes an argument, and may occur
 * multiple times
 */
fn optmulti(name: ~str) -> opt {
    return {name: mkname(name), hasarg: yes, occur: multi};
}

enum optval { val(~str), given, }

/**
 * The result of checking command line arguments. Contains a vector
 * of matches and a vector of free strings.
 */
type matches = {opts: ~[opt], vals: ~[~[optval]], free: ~[~str]};

fn is_arg(arg: ~str) -> bool {
    return str::len(arg) > 1u && arg[0] == '-' as u8;
}

fn name_str(nm: name) -> ~str {
    return match nm {
      short(ch) => str::from_char(ch),
      long(s) => s
    };
}

fn find_opt(opts: ~[opt], nm: name) -> Option<uint> {
    vec::position(opts, |opt| opt.name == nm)
}

/**
 * The type returned when the command line does not conform to the
 * expected format. Pass this value to <fail_str> to get an error message.
 */
enum fail_ {
    argument_missing(~str),
    unrecognized_option(~str),
    option_missing(~str),
    option_duplicated(~str),
    unexpected_argument(~str),
}

/// Convert a `fail_` enum into an error string
fn fail_str(f: fail_) -> ~str {
    return match f {
      argument_missing(nm) => ~"Argument to option '" + nm + ~"' missing.",
      unrecognized_option(nm) => ~"Unrecognized option: '" + nm + ~"'.",
      option_missing(nm) => ~"Required option '" + nm + ~"' missing.",
      option_duplicated(nm) => ~"Option '" + nm + ~"' given more than once.",
      unexpected_argument(nm) => {
        ~"Option " + nm + ~" does not take an argument."
      }
    };
}

/**
 * The result of parsing a command line with a set of options
 * (result::t<matches, fail_>)
 */
type result = result::result<matches, fail_>;

/**
 * Parse command line arguments according to the provided options
 *
 * On success returns `ok(opt)`. Use functions such as `opt_present`
 * `opt_str`, etc. to interrogate results.  Returns `err(fail_)` on failure.
 * Use <fail_str> to get an error message.
 */
fn getopts(args: ~[~str], opts: ~[opt]) -> result unsafe {
    let n_opts = vec::len::<opt>(opts);
    fn f(_x: uint) -> ~[optval] { return ~[]; }
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
                    names = ~[long(tail)];
                } else {
                    names =
                        ~[long(tail_eq[0])];
                    i_arg =
                        option::Some::<~str>(tail_eq[1]);
                }
            } else {
                let mut j = 1u;
                let mut last_valid_opt_id = option::None;
                names = ~[];
                while j < curlen {
                    let range = str::char_range_at(cur, j);
                    let opt = short(range.ch);

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

                              yes | maybe => true,
                              no => false
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
                let optid = match find_opt(opts, nm) {
                  Some(id) => id,
                  None => return err(unrecognized_option(name_str(nm)))
                };
                match opts[optid].hasarg {
                  no => {
                    if !option::is_none::<~str>(i_arg) {
                        return err(unexpected_argument(name_str(nm)));
                    }
                    vec::push(vals[optid], given);
                  }
                  maybe => {
                    if !option::is_none::<~str>(i_arg) {
                        vec::push(vals[optid], val(option::get(i_arg)));
                    } else if name_pos < vec::len::<name>(names) ||
                                  i + 1u == l || is_arg(args[i + 1u]) {
                        vec::push(vals[optid], given);
                    } else { i += 1u; vec::push(vals[optid], val(args[i])); }
                  }
                  yes => {
                    if !option::is_none::<~str>(i_arg) {
                        vec::push(vals[optid],
                                  val(option::get::<~str>(i_arg)));
                    } else if i + 1u == l {
                        return err(argument_missing(name_str(nm)));
                    } else { i += 1u; vec::push(vals[optid], val(args[i])); }
                  }
                }
            }
        }
        i += 1u;
    }
    i = 0u;
    while i < n_opts {
        let n = vec::len::<optval>(vals[i]);
        let occ = opts[i].occur;
        if occ == req {
            if n == 0u {
                return err(option_missing(name_str(opts[i].name)));
            }
        }
        if occ != multi {
            if n > 1u {
                return err(option_duplicated(name_str(opts[i].name)));
            }
        }
        i += 1u;
    }
    return ok({opts: opts, vals: vec::from_mut(vals), free: free});
}

fn opt_vals(m: matches, nm: ~str) -> ~[optval] {
    return match find_opt(m.opts, mkname(nm)) {
      Some(id) => m.vals[id],
      None => {
        error!("No option '%s' defined", nm);
        fail
      }
    };
}

fn opt_val(m: matches, nm: ~str) -> optval { return opt_vals(m, nm)[0]; }

/// Returns true if an option was matched
fn opt_present(m: matches, nm: ~str) -> bool {
    return vec::len::<optval>(opt_vals(m, nm)) > 0u;
}

/// Returns true if any of several options were matched
fn opts_present(m: matches, names: ~[~str]) -> bool {
    for vec::each(names) |nm| {
        match find_opt(m.opts, mkname(nm)) {
          Some(_) => return true,
          _ => ()
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
fn opt_str(m: matches, nm: ~str) -> ~str {
    return match opt_val(m, nm) { val(s) => s, _ => fail };
}

/**
 * Returns the string argument supplied to one of several matching options
 *
 * Fails if the no option was provided from the given list, or if the no such
 * option took an argument
 */
fn opts_str(m: matches, names: ~[~str]) -> ~str {
    for vec::each(names) |nm| {
        match opt_val(m, nm) {
          val(s) => return s,
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
fn opt_strs(m: matches, nm: ~str) -> ~[~str] {
    let mut acc: ~[~str] = ~[];
    for vec::each(opt_vals(m, nm)) |v| {
        match v { val(s) => vec::push(acc, s), _ => () }
    }
    return acc;
}

/// Returns the string argument supplied to a matching option or none
fn opt_maybe_str(m: matches, nm: ~str) -> Option<~str> {
    let vals = opt_vals(m, nm);
    if vec::len::<optval>(vals) == 0u { return None::<~str>; }
    return match vals[0] { val(s) => Some::<~str>(s), _ => None::<~str> };
}


/**
 * Returns the matching string, a default, or none
 *
 * Returns none if the option was not present, `def` if the option was
 * present but no argument was provided, and the argument if the option was
 * present and an argument was provided.
 */
fn opt_default(m: matches, nm: ~str, def: ~str) -> Option<~str> {
    let vals = opt_vals(m, nm);
    if vec::len::<optval>(vals) == 0u { return None::<~str>; }
    return match vals[0] { val(s) => Some::<~str>(s), _ => Some::<~str>(def) }
}

#[cfg(test)]
mod tests {
    import opt = getopts;
    import result::{err, ok};

    enum fail_type {
        argument_missing_,
        unrecognized_option_,
        option_missing_,
        option_duplicated_,
        unexpected_argument_,
    }

    fn check_fail_type(f: fail_, ft: fail_type) {
        match f {
          argument_missing(_) => assert ft == argument_missing_,
          unrecognized_option(_) => assert ft == unrecognized_option_,
          option_missing(_) => assert ft == option_missing_,
          option_duplicated(_) => assert ft == option_duplicated_,
          unexpected_argument(_) => assert ft == unexpected_argument_
        }
    }


    // Tests for reqopt
    #[test]
    fn test_reqopt_long() {
        let args = ~[~"--test=20"];
        let opts = ~[reqopt(~"test")];
        let rs = getopts(args, opts);
        match rs {
          ok(m) => {
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
          err(f) => check_fail_type(f, option_missing_),
          _ => fail
        }
    }

    #[test]
    fn test_reqopt_long_no_arg() {
        let args = ~[~"--test"];
        let opts = ~[reqopt(~"test")];
        let rs = getopts(args, opts);
        match rs {
          err(f) => check_fail_type(f, argument_missing_),
          _ => fail
        }
    }

    #[test]
    fn test_reqopt_long_multi() {
        let args = ~[~"--test=20", ~"--test=30"];
        let opts = ~[reqopt(~"test")];
        let rs = getopts(args, opts);
        match rs {
          err(f) => check_fail_type(f, option_duplicated_),
          _ => fail
        }
    }

    #[test]
    fn test_reqopt_short() {
        let args = ~[~"-t", ~"20"];
        let opts = ~[reqopt(~"t")];
        let rs = getopts(args, opts);
        match rs {
          ok(m) => {
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
          err(f) => check_fail_type(f, option_missing_),
          _ => fail
        }
    }

    #[test]
    fn test_reqopt_short_no_arg() {
        let args = ~[~"-t"];
        let opts = ~[reqopt(~"t")];
        let rs = getopts(args, opts);
        match rs {
          err(f) => check_fail_type(f, argument_missing_),
          _ => fail
        }
    }

    #[test]
    fn test_reqopt_short_multi() {
        let args = ~[~"-t", ~"20", ~"-t", ~"30"];
        let opts = ~[reqopt(~"t")];
        let rs = getopts(args, opts);
        match rs {
          err(f) => check_fail_type(f, option_duplicated_),
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
          ok(m) => {
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
          ok(m) => assert (!opt_present(m, ~"test")),
          _ => fail
        }
    }

    #[test]
    fn test_optopt_long_no_arg() {
        let args = ~[~"--test"];
        let opts = ~[optopt(~"test")];
        let rs = getopts(args, opts);
        match rs {
          err(f) => check_fail_type(f, argument_missing_),
          _ => fail
        }
    }

    #[test]
    fn test_optopt_long_multi() {
        let args = ~[~"--test=20", ~"--test=30"];
        let opts = ~[optopt(~"test")];
        let rs = getopts(args, opts);
        match rs {
          err(f) => check_fail_type(f, option_duplicated_),
          _ => fail
        }
    }

    #[test]
    fn test_optopt_short() {
        let args = ~[~"-t", ~"20"];
        let opts = ~[optopt(~"t")];
        let rs = getopts(args, opts);
        match rs {
          ok(m) => {
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
          ok(m) => assert (!opt_present(m, ~"t")),
          _ => fail
        }
    }

    #[test]
    fn test_optopt_short_no_arg() {
        let args = ~[~"-t"];
        let opts = ~[optopt(~"t")];
        let rs = getopts(args, opts);
        match rs {
          err(f) => check_fail_type(f, argument_missing_),
          _ => fail
        }
    }

    #[test]
    fn test_optopt_short_multi() {
        let args = ~[~"-t", ~"20", ~"-t", ~"30"];
        let opts = ~[optopt(~"t")];
        let rs = getopts(args, opts);
        match rs {
          err(f) => check_fail_type(f, option_duplicated_),
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
          ok(m) => assert (opt_present(m, ~"test")),
          _ => fail
        }
    }

    #[test]
    fn test_optflag_long_missing() {
        let args = ~[~"blah"];
        let opts = ~[optflag(~"test")];
        let rs = getopts(args, opts);
        match rs {
          ok(m) => assert (!opt_present(m, ~"test")),
          _ => fail
        }
    }

    #[test]
    fn test_optflag_long_arg() {
        let args = ~[~"--test=20"];
        let opts = ~[optflag(~"test")];
        let rs = getopts(args, opts);
        match rs {
          err(f) => {
            log(error, fail_str(f));
            check_fail_type(f, unexpected_argument_);
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
          err(f) => check_fail_type(f, option_duplicated_),
          _ => fail
        }
    }

    #[test]
    fn test_optflag_short() {
        let args = ~[~"-t"];
        let opts = ~[optflag(~"t")];
        let rs = getopts(args, opts);
        match rs {
          ok(m) => assert (opt_present(m, ~"t")),
          _ => fail
        }
    }

    #[test]
    fn test_optflag_short_missing() {
        let args = ~[~"blah"];
        let opts = ~[optflag(~"t")];
        let rs = getopts(args, opts);
        match rs {
          ok(m) => assert (!opt_present(m, ~"t")),
          _ => fail
        }
    }

    #[test]
    fn test_optflag_short_arg() {
        let args = ~[~"-t", ~"20"];
        let opts = ~[optflag(~"t")];
        let rs = getopts(args, opts);
        match rs {
          ok(m) => {
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
          err(f) => check_fail_type(f, option_duplicated_),
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
          ok(m) => {
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
          ok(m) => assert (!opt_present(m, ~"test")),
          _ => fail
        }
    }

    #[test]
    fn test_optmulti_long_no_arg() {
        let args = ~[~"--test"];
        let opts = ~[optmulti(~"test")];
        let rs = getopts(args, opts);
        match rs {
          err(f) => check_fail_type(f, argument_missing_),
          _ => fail
        }
    }

    #[test]
    fn test_optmulti_long_multi() {
        let args = ~[~"--test=20", ~"--test=30"];
        let opts = ~[optmulti(~"test")];
        let rs = getopts(args, opts);
        match rs {
          ok(m) => {
            assert (opt_present(m, ~"test"));
            assert (opt_str(m, ~"test") == ~"20");
            assert (opt_strs(m, ~"test")[0] == ~"20");
            assert (opt_strs(m, ~"test")[1] == ~"30");
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
          ok(m) => {
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
          ok(m) => assert (!opt_present(m, ~"t")),
          _ => fail
        }
    }

    #[test]
    fn test_optmulti_short_no_arg() {
        let args = ~[~"-t"];
        let opts = ~[optmulti(~"t")];
        let rs = getopts(args, opts);
        match rs {
          err(f) => check_fail_type(f, argument_missing_),
          _ => fail
        }
    }

    #[test]
    fn test_optmulti_short_multi() {
        let args = ~[~"-t", ~"20", ~"-t", ~"30"];
        let opts = ~[optmulti(~"t")];
        let rs = getopts(args, opts);
        match rs {
          ok(m) => {
            assert (opt_present(m, ~"t"));
            assert (opt_str(m, ~"t") == ~"20");
            assert (opt_strs(m, ~"t")[0] == ~"20");
            assert (opt_strs(m, ~"t")[1] == ~"30");
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
          err(f) => check_fail_type(f, unrecognized_option_),
          _ => fail
        }
    }

    #[test]
    fn test_unrecognized_option_short() {
        let args = ~[~"-t"];
        let opts = ~[optmulti(~"test")];
        let rs = getopts(args, opts);
        match rs {
          err(f) => check_fail_type(f, unrecognized_option_),
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
          ok(m) => {
            assert (m.free[0] == ~"prog");
            assert (m.free[1] == ~"free1");
            assert (opt_str(m, ~"s") == ~"20");
            assert (m.free[2] == ~"free2");
            assert (opt_present(m, ~"flag"));
            assert (opt_str(m, ~"long") == ~"30");
            assert (opt_present(m, ~"f"));
            assert (opt_strs(m, ~"m")[0] == ~"40");
            assert (opt_strs(m, ~"m")[1] == ~"50");
            assert (opt_strs(m, ~"n")[0] == ~"-A B");
            assert (opt_strs(m, ~"n")[1] == ~"-60 70");
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
          result::ok(m) => m,
          result::err(_) => fail
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
          result::ok(m) => m,
          result::err(_) => fail
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
