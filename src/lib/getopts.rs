

/* Simple getopt alternative. Construct a vector of options, either by using
 * reqopt, optopt, and optflag or by building them from components yourself,
 * and pass them to getopts, along with a vector of actual arguments (not
 * including argv[0]). You'll either get a failure code back, or a match.
 * You'll have to verify whether the amount of 'free' arguments in the match
 * is what you expect. Use opt_* accessors (bottom of the file) to get
 * argument values out of the match object.
 */
import option::some;
import option::none;
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

tag name { long(istr); short(char); }

tag hasarg { yes; no; maybe; }

tag occur { req; optional; multi; }

type opt = {name: name, hasarg: hasarg, occur: occur};

fn mkname(nm: &istr) -> name {
    ret if str::char_len(nm) == 1u {
            short(str::char_at(nm, 0u))
        } else { long(nm) };
}

fn reqopt(name: &istr) -> opt {
    ret {name: mkname(name), hasarg: yes, occur: req};
}

fn optopt(name: &istr) -> opt {
    ret {name: mkname(name), hasarg: yes, occur: optional};
}

fn optflag(name: &istr) -> opt {
    ret {name: mkname(name), hasarg: no, occur: optional};
}

fn optflagopt(name: &istr) -> opt {
    ret {name: mkname(name), hasarg: maybe, occur: optional};
}

fn optmulti(name: &istr) -> opt {
    ret {name: mkname(name), hasarg: yes, occur: multi};
}

tag optval { val(istr); given; }

type match = {opts: [opt], vals: [mutable [optval]], free: [istr]};

fn is_arg(arg: &istr) -> bool {
    ret str::byte_len(arg) > 1u && arg[0] == '-' as u8;
}

fn name_str(nm: &name) -> istr {
    ret alt nm { short(ch) { str::from_char(ch) } long(s) { s } };
}

fn find_opt(opts: &[opt], nm: &name) -> option::t<uint> {
    let i = 0u;
    let l = vec::len::<opt>(opts);
    while i < l { if opts[i].name == nm { ret some::<uint>(i); } i += 1u; }
    ret none::<uint>;
}

tag fail_ {
    argument_missing(istr);
    unrecognized_option(istr);
    option_missing(istr);
    option_duplicated(istr);
    unexpected_argument(istr);
}

fn fail_str(f: &fail_) -> istr {
    ret alt f {
          argument_missing(nm) {
            ~"Argument to option '" + nm + ~"' missing." }
          unrecognized_option(nm) {
            ~"Unrecognized option: '" + nm + ~"'." }
          option_missing(nm) { ~"Required option '" + nm + ~"' missing." }
          option_duplicated(nm) {
            ~"Option '" + nm + ~"' given more than once."
          }
          unexpected_argument(nm) {
            ~"Option " + nm + ~" does not take an argument."
          }
        };
}

tag result { success(match); failure(fail_); }

fn getopts(args: &[istr], opts: &[opt]) -> result {
    let n_opts = vec::len::<opt>(opts);
    fn f(_x: uint) -> [optval] { ret []; }
    let vals = vec::init_fn_mut::<[optval]>(f, n_opts);
    let free: [istr] = [];
    let l = vec::len(args);
    let i = 0u;
    while i < l {
        let cur = args[i];
        let curlen = str::byte_len(cur);
        if !is_arg(cur) {
            free += [cur];
        } else if str::eq(cur, ~"--") {
            let j = i + 1u;
            while j < l { free += [args[j]]; j += 1u; }
            break;
        } else {
            let names;
            let i_arg = option::none::<istr>;
            if cur[1] == '-' as u8 {
                let tail = str::slice(cur, 2u, curlen);
                let eq = str::index(tail, '=' as u8);
                if eq == -1 {
                    names = [long(tail)];
                } else {
                    names = [long(str::slice(tail, 0u, eq as uint))];
                    i_arg =
                        option::some::<istr>(str::slice(tail,
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
                    if !option::is_none::<istr>(i_arg) {
                        ret failure(unexpected_argument(name_str(nm)));
                    }
                    vals[optid] += [given];
                  }
                  maybe. {
                    if !option::is_none::<istr>(i_arg) {
                        vals[optid] += [val(option::get(i_arg))];
                    } else if name_pos < vec::len::<name>(names) ||
                                  i + 1u == l || is_arg(args[i + 1u]) {
                        vals[optid] += [given];
                    } else { i += 1u; vals[optid] += [val(args[i])]; }
                  }
                  yes. {
                    if !option::is_none::<istr>(i_arg) {
                        vals[optid] += [val(option::get::<istr>(i_arg))];
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

fn opt_vals(m: &match, nm: &istr) -> [optval] {
    ret alt find_opt(m.opts, mkname(nm)) {
          some(id) { m.vals[id] }
          none. { log_err ~"No option '" + nm + ~"' defined."; fail }
        };
}

fn opt_val(m: &match, nm: &istr) -> optval { ret opt_vals(m, nm)[0]; }

fn opt_present(m: &match, nm: &istr) -> bool {
    ret vec::len::<optval>(opt_vals(m, nm)) > 0u;
}

fn opt_str(m: &match, nm: &istr) -> istr {
    ret alt opt_val(m, nm) { val(s) { s } _ { fail } };
}

fn opt_strs(m: &match, nm: &istr) -> [istr] {
    let acc: [istr] = [];
    for v: optval in opt_vals(m, nm) {
        alt v { val(s) { acc += [s]; } _ { } }
    }
    ret acc;
}

fn opt_maybe_str(m: &match, nm: &istr) -> option::t<istr> {
    let vals = opt_vals(m, nm);
    if vec::len::<optval>(vals) == 0u { ret none::<istr>; }
    ret alt vals[0] { val(s) { some::<istr>(s) } _ { none::<istr> } };
}


/// Returns none if the option was not present, `def` if the option was
/// present but no argument was provided, and the argument if the option was
/// present and an argument was provided.
fn opt_default(m: &match, nm: &istr, def: &istr) -> option::t<istr> {
    let vals = opt_vals(m, nm);
    if vec::len::<optval>(vals) == 0u { ret none::<istr>; }
    ret alt vals[0] { val(s) { some::<istr>(s) } _ { some::<istr>(def) } }
}
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
