

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

tag name { long(str); short(char); }

tag hasarg { yes; no; maybe; }

tag occur { req; optional; multi; }

type opt = rec(name name, hasarg hasarg, occur occur);

fn mkname(str nm) -> name {
    ret if (str::char_len(nm) == 1u) {
            short(str::char_at(nm, 0u))
        } else { long(nm) };
}

fn reqopt(str name) -> opt {
    ret rec(name=mkname(name), hasarg=yes, occur=req);
}

fn optopt(str name) -> opt {
    ret rec(name=mkname(name), hasarg=yes, occur=optional);
}

fn optflag(str name) -> opt {
    ret rec(name=mkname(name), hasarg=no, occur=optional);
}

fn optflagopt(str name) -> opt {
    ret rec(name=mkname(name), hasarg=maybe, occur=optional);
}

fn optmulti(str name) -> opt {
    ret rec(name=mkname(name), hasarg=yes, occur=multi);
}

tag optval { val(str); given; }

type match = rec(vec[opt] opts, vec[mutable vec[optval]] vals, vec[str] free);

fn is_arg(str arg) -> bool {
    ret str::byte_len(arg) > 1u && arg.(0) == '-' as u8;
}

fn name_str(name nm) -> str {
    ret alt (nm) {
            case (short(?ch)) { str::from_char(ch) }
            case (long(?s)) { s }
        };
}


// FIXME rustboot workaround
fn name_eq(name a, name b) -> bool {
    ret alt (a) {
            case (long(?a)) {
                alt (b) {
                    case (long(?b)) { str::eq(a, b) }
                    case (_) { false }
                }
            }
            case (_) { if (a == b) { true } else { false } }
        };
}

fn find_opt(vec[opt] opts, name nm) -> option::t[uint] {
    auto i = 0u;
    auto l = vec::len[opt](opts);
    while (i < l) {
        if (name_eq(opts.(i).name, nm)) { ret some[uint](i); }
        i += 1u;
    }
    ret none[uint];
}

tag fail_ {
    argument_missing(str);
    unrecognized_option(str);
    option_missing(str);
    option_duplicated(str);
    unexpected_argument(str);
}

fn fail_str(fail_ f) -> str {
    ret alt (f) {
            case (argument_missing(?nm)) {
                "Argument to option '" + nm + "' missing."
            }
            case (unrecognized_option(?nm)) {
                "Unrecognized option: '" + nm + "'."
            }
            case (option_missing(?nm)) {
                "Required option '" + nm + "' missing."
            }
            case (option_duplicated(?nm)) {
                "Option '" + nm + "' given more than once."
            }
            case (unexpected_argument(?nm)) {
                "Option " + nm + " does not take an argument."
            }
        };
}

tag result { success(match); failure(fail_); }

fn getopts(vec[str] args, vec[opt] opts) -> result {
    auto n_opts = vec::len[opt](opts);
    fn empty_(uint x) -> vec[optval] { ret vec::empty[optval](); }
    auto f = empty_;
    auto vals = vec::init_fn_mut[vec[optval]](f, n_opts);
    let vec[str] free = [];
    auto l = vec::len[str](args);
    auto i = 0u;
    while (i < l) {
        auto cur = args.(i);
        auto curlen = str::byte_len(cur);
        if (!is_arg(cur)) {
            vec::push[str](free, cur);
        } else if (str::eq(cur, "--")) {
            free += vec::slice[str](args, i + 1u, l);
            break;
        } else {
            auto names;
            auto i_arg = option::none[str];
            if (cur.(1) == '-' as u8) {
                auto tail = str::slice(cur, 2u, curlen);
                auto eq = str::index(tail, '=' as u8);
                if (eq == -1) {
                    names = [long(tail)];
                } else {
                    names = [long(str::slice(tail, 0u, eq as uint))];
                    i_arg =
                        option::some[str](str::slice(tail, (eq as uint) + 1u,
                                                     curlen - 2u));
                }
            } else {
                auto j = 1u;
                names = [];
                while (j < curlen) {
                    auto range = str::char_range_at(cur, j);
                    vec::push[name](names, short(range._0));
                    j = range._1;
                }
            }
            auto name_pos = 0u;
            for (name nm in names) {
                name_pos += 1u;
                auto optid;
                alt (find_opt(opts, nm)) {
                    case (some(?id)) { optid = id; }
                    case (none) {
                        ret failure(unrecognized_option(name_str(nm)));
                    }
                }
                alt (opts.(optid).hasarg) {
                    case (no) {
                        if (!option::is_none[str](i_arg)) {
                            ret failure(unexpected_argument(name_str(nm)));
                        }
                        vec::push[optval](vals.(optid), given);
                    }
                    case (maybe) {
                        if (!option::is_none[str](i_arg)) {
                            vec::push[optval](vals.(optid),
                                              val(option::get[str](i_arg)));
                        } else if (name_pos < vec::len[name](names) ||
                                       i + 1u == l || is_arg(args.(i + 1u))) {
                            vec::push[optval](vals.(optid), given);
                        } else {
                            i += 1u;
                            vec::push[optval](vals.(optid), val(args.(i)));
                        }
                    }
                    case (yes) {
                        if (!option::is_none[str](i_arg)) {
                            vec::push[optval](vals.(optid),
                                              val(option::get[str](i_arg)));
                        } else if (i + 1u == l) {
                            ret failure(argument_missing(name_str(nm)));
                        } else {
                            i += 1u;
                            vec::push[optval](vals.(optid), val(args.(i)));
                        }
                    }
                }
            }
        }
        i += 1u;
    }
    i = 0u;
    while (i < n_opts) {
        auto n = vec::len[optval](vals.(i));
        auto occ = opts.(i).occur;
        if (occ == req) {
            if (n == 0u) {
                ret failure(option_missing(name_str(opts.(i).name)));
            }
        }
        if (occ != multi) {
            if (n > 1u) {
                ret failure(option_duplicated(name_str(opts.(i).name)));
            }
        }
        i += 1u;
    }
    ret success(rec(opts=opts, vals=vals, free=free));
}

fn opt_vals(match m, str nm) -> vec[optval] {
    ret alt (find_opt(m.opts, mkname(nm))) {
            case (some(?id)) { m.vals.(id) }
            case (none) { log_err "No option '" + nm + "' defined."; fail }
        };
}

fn opt_val(match m, str nm) -> optval { ret opt_vals(m, nm).(0); }

fn opt_present(match m, str nm) -> bool {
    ret vec::len[optval](opt_vals(m, nm)) > 0u;
}

fn opt_str(match m, str nm) -> str {
    ret alt (opt_val(m, nm)) { case (val(?s)) { s } case (_) { fail } };
}

fn opt_strs(match m, str nm) -> vec[str] {
    let vec[str] acc = [];
    for (optval v in opt_vals(m, nm)) {
        alt (v) { case (val(?s)) { vec::push[str](acc, s); } case (_) { } }
    }
    ret acc;
}

fn opt_maybe_str(match m, str nm) -> option::t[str] {
    auto vals = opt_vals(m, nm);
    if (vec::len[optval](vals) == 0u) { ret none[str]; }
    ret alt (vals.(0)) {
            case (val(?s)) { some[str](s) }
            case (_) { none[str] }
        };
}


/// Returns none if the option was not present, `def` if the option was
/// present but no argument was provided, and the argument if the option was
/// present and an argument was provided.
fn opt_default(match m, str nm, str def) -> option::t[str] {
    auto vals = opt_vals(m, nm);
    if (vec::len[optval](vals) == 0u) { ret none[str]; }
    ret alt (vals.(0)) {
            case (val(?s)) { some[str](s) }
            case (_) { some[str](def) }
        }
}
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
