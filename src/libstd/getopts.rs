/*
Module: getopts

Simple getopt alternative. Construct a vector of options, either by using
reqopt, optopt, and optflag or by building them from components yourself, and
pass them to getopts, along with a vector of actual arguments (not including
argv[0]). You'll either get a failure code back, or a match.  You'll have to
verify whether the amount of 'free' arguments in the match is what you
expect. Use opt_* accessors to get argument values out of the match object.

Single-character options are expected to appear on the command line with a
single preceeding dash; multiple-character options are expected to be
proceeded by two dashes. Options that expect an argument accept their argument
following either a space or an equals sign.

Example:

The following example shows simple command line parsing for an application
that requires an input file to be specified, accepts an optional output file
name following -o, and accepts both -h and --help as optional flags.

> fn main(args: [str]) {
>   let opts = [
>     optopt("o"),
>     optflag("h"),
>     optflag("help")
>   ];
>   let match = alt getopts(vec::shift(args), opts) {
>     success(m) { m }
>     failure(f) { fail fail_str(f) }
>   };
>   if opt_present(match, "h") || opt_present(match, "help") {
>     print_usage();
>     ret;
>   }
>   let output = opt_maybe_str(match, "o");
>   let input = if !vec::is_empty(match.free) {
>     match.free[0]
>   } else {
>     print_usage();
>     ret;
>   }
>   do_work(input, output);
> }

*/

import core::option;
import option::{some, none};
export opt;
export reqopt;
export optopt;
export optflag;
export optflagopt;
export optmulti;
export getopts;
export result;
export success;
export failure;
export match;
export fail_;
export fail_str;
export opt_present;
export opt_str;
export opt_strs;
export opt_maybe_str;
export opt_default;

tag name { long(str); short(char); }

tag hasarg { yes; no; maybe; }

tag occur { req; optional; multi; }

/*
Type: opt

A description of a possible option
*/
type opt = {name: name, hasarg: hasarg, occur: occur};

fn mkname(nm: str) -> name {
    ret if str::char_len(nm) == 1u {
            short(str::char_at(nm, 0u))
        } else { long(nm) };
}

/*
Function: reqopt

Create an option that is required and takes an argument
*/
fn reqopt(name: str) -> opt {
    ret {name: mkname(name), hasarg: yes, occur: req};
}

/*
Function: optopt

Create an option that is optional and takes an argument
*/
fn optopt(name: str) -> opt {
    ret {name: mkname(name), hasarg: yes, occur: optional};
}

/*
Function: optflag

Create an option that is optional and does not take an argument
*/
fn optflag(name: str) -> opt {
    ret {name: mkname(name), hasarg: no, occur: optional};
}

/*
Function: optflagopt

Create an option that is optional and takes an optional argument
*/
fn optflagopt(name: str) -> opt {
    ret {name: mkname(name), hasarg: maybe, occur: optional};
}

/*
Function: optmulti

Create an option that is optional, takes an argument, and may occur
multiple times
*/
fn optmulti(name: str) -> opt {
    ret {name: mkname(name), hasarg: yes, occur: multi};
}

tag optval { val(str); given; }

/*
Type: match

The result of checking command line arguments. Contains a vector
of matches and a vector of free strings.
*/
type match = {opts: [opt], vals: [mutable [optval]], free: [str]};

fn is_arg(arg: str) -> bool {
    ret str::byte_len(arg) > 1u && arg[0] == '-' as u8;
}

fn name_str(nm: name) -> str {
    ret alt nm { short(ch) { str::from_char(ch) } long(s) { s } };
}

fn find_opt(opts: [opt], nm: name) -> option::t<uint> {
    let i = 0u;
    let l = vec::len::<opt>(opts);
    while i < l { if opts[i].name == nm { ret some::<uint>(i); } i += 1u; }
    ret none::<uint>;
}

/*
Type: fail_

The type returned when the command line does not conform to the
expected format. Pass this value to <fail_str> to get an error message.
*/
tag fail_ {
    argument_missing(str);
    unrecognized_option(str);
    option_missing(str);
    option_duplicated(str);
    unexpected_argument(str);
}

/*
Function: fail_str

Convert a <fail_> tag into an error string
*/
fn fail_str(f: fail_) -> str {
    ret alt f {
          argument_missing(nm) { "Argument to option '" + nm + "' missing." }
          unrecognized_option(nm) { "Unrecognized option: '" + nm + "'." }
          option_missing(nm) { "Required option '" + nm + "' missing." }
          option_duplicated(nm) {
            "Option '" + nm + "' given more than once."
          }
          unexpected_argument(nm) {
            "Option " + nm + " does not take an argument."
          }
        };
}

/*
Type: result

The result of parsing a command line with a set of options

Variants:

success(match) - Returned from getopts on success
failure(fail_) - Returned from getopts on failure
*/
tag result { success(match); failure(fail_); }

/*
Function: getopts

Parse command line arguments according to the provided options

Returns:

success(match) - On success. Use functions such as <opt_present>
                 <opt_str>, etc. to interrogate results.
failure(fail_) - On failure. Use <fail_str> to get an error message.
*/
fn getopts(args: [str], opts: [opt]) -> result {
    let n_opts = vec::len::<opt>(opts);
    fn f(_x: uint) -> [optval] { ret []; }
    let vals = vec::init_fn_mut::<[optval]>(f, n_opts);
    let free: [str] = [];
    let l = vec::len(args);
    let i = 0u;
    while i < l {
        let cur = args[i];
        let curlen = str::byte_len(cur);
        if !is_arg(cur) {
            free += [cur];
        } else if str::eq(cur, "--") {
            let j = i + 1u;
            while j < l { free += [args[j]]; j += 1u; }
            break;
        } else {
            let names;
            let i_arg = option::none::<str>;
            if cur[1] == '-' as u8 {
                let tail = str::slice(cur, 2u, curlen);
                let eq = str::index(tail, '=' as u8);
                if eq == -1 {
                    names = [long(tail)];
                } else {
                    names = [long(str::slice(tail, 0u, eq as uint))];
                    i_arg =
                        option::some::<str>(str::slice(tail,
                                                       (eq as uint) + 1u,
                                                       curlen - 2u));
                }
            } else {
                let j = 1u;
                names = [];
                while j < curlen {
                    let range = str::char_range_at(cur, j);
                    names += [short(range.ch)];
                    j = range.next;
                }
            }
            let name_pos = 0u;
            for nm: name in names {
                name_pos += 1u;
                let optid;
                alt find_opt(opts, nm) {
                  some(id) { optid = id; }
                  none. { ret failure(unrecognized_option(name_str(nm))); }
                }
                alt opts[optid].hasarg {
                  no. {
                    if !option::is_none::<str>(i_arg) {
                        ret failure(unexpected_argument(name_str(nm)));
                    }
                    vals[optid] += [given];
                  }
                  maybe. {
                    if !option::is_none::<str>(i_arg) {
                        vals[optid] += [val(option::get(i_arg))];
                    } else if name_pos < vec::len::<name>(names) ||
                                  i + 1u == l || is_arg(args[i + 1u]) {
                        vals[optid] += [given];
                    } else { i += 1u; vals[optid] += [val(args[i])]; }
                  }
                  yes. {
                    if !option::is_none::<str>(i_arg) {
                        vals[optid] += [val(option::get::<str>(i_arg))];
                    } else if i + 1u == l {
                        ret failure(argument_missing(name_str(nm)));
                    } else { i += 1u; vals[optid] += [val(args[i])]; }
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
                ret failure(option_missing(name_str(opts[i].name)));
            }
        }
        if occ != multi {
            if n > 1u {
                ret failure(option_duplicated(name_str(opts[i].name)));
            }
        }
        i += 1u;
    }
    ret success({opts: opts, vals: vals, free: free});
}

fn opt_vals(m: match, nm: str) -> [optval] {
    ret alt find_opt(m.opts, mkname(nm)) {
          some(id) { m.vals[id] }
          none. { log_err "No option '" + nm + "' defined."; fail }
        };
}

fn opt_val(m: match, nm: str) -> optval { ret opt_vals(m, nm)[0]; }

/*
Function: opt_present

Returns true if an option was matched
*/
fn opt_present(m: match, nm: str) -> bool {
    ret vec::len::<optval>(opt_vals(m, nm)) > 0u;
}

/*
Function: opt_str

Returns the string argument supplied to a matching option

Failure:

- If the option was not matched
- If the match did not take an argument
*/
fn opt_str(m: match, nm: str) -> str {
    ret alt opt_val(m, nm) { val(s) { s } _ { fail } };
}

/*
Function: opt_str

Returns a vector of the arguments provided to all matches of the given option.
Used when an option accepts multiple values.
*/
fn opt_strs(m: match, nm: str) -> [str] {
    let acc: [str] = [];
    for v: optval in opt_vals(m, nm) {
        alt v { val(s) { acc += [s]; } _ { } }
    }
    ret acc;
}

/*
Function: opt_str

Returns the string argument supplied to a matching option or none
*/
fn opt_maybe_str(m: match, nm: str) -> option::t<str> {
    let vals = opt_vals(m, nm);
    if vec::len::<optval>(vals) == 0u { ret none::<str>; }
    ret alt vals[0] { val(s) { some::<str>(s) } _ { none::<str> } };
}


/*
Function: opt_default

Returns the matching string, a default, or none

Returns none if the option was not present, `def` if the option was
present but no argument was provided, and the argument if the option was
present and an argument was provided.
*/
fn opt_default(m: match, nm: str, def: str) -> option::t<str> {
    let vals = opt_vals(m, nm);
    if vec::len::<optval>(vals) == 0u { ret none::<str>; }
    ret alt vals[0] { val(s) { some::<str>(s) } _ { some::<str>(def) } }
}
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
