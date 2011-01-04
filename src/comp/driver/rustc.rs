// -*- rust -*-

import front.parser;
import front.token;
import middle.trans;
import middle.resolve;
import middle.typeck;
import util.common;

import std.option;
import std.option.some;
import std.option.none;
import std._str;
import std._vec;

impure fn parse_input(session.session sess,
                      parser.parser p,
                      str input) -> @front.ast.crate {
    if (_str.ends_with(input, ".rc")) {
        ret parser.parse_crate_from_crate_file(p);
    } else if (_str.ends_with(input, ".rs")) {
        ret parser.parse_crate_from_source_file(p);
    }
    sess.err("unknown unput file type: " + input);
    fail;
}

impure fn compile_input(session.session sess, str input, str output,
                        bool shared) {
    auto p = parser.new_parser(sess, 0, input);
    auto crate = parse_input(sess, p, input);
    crate = resolve.resolve_crate(sess, crate);
    crate = typeck.check_crate(sess, crate);
    trans.trans_crate(sess, crate, output, shared);
}

fn warn_wrong_compiler() {
    log "This is the rust 'self-hosted' compiler.";
    log "The one written in rust.";
    log "It does nothing yet, it's a placeholder.";
    log "You want rustboot, the compiler next door.";
}

fn usage(session.session sess, str argv0) {
    log #fmt("usage: %s [options] <input>", argv0);
    log "options:";
    log "";
    log "    -o <filename>      write output to <filename>";
    log "    -nowarn            suppress wrong-compiler warning";
    log "    -shared            compile a shared-library crate";
    log "    -h                 display this message";
    log "";
    log "";
}

fn get_os() -> session.os {
    auto s = std.os.target_os();
    if (_str.eq(s, "win32")) { ret session.os_win32; }
    if (_str.eq(s, "macos")) { ret session.os_macos; }
    if (_str.eq(s, "linux")) { ret session.os_linux; }
}

impure fn main(vec[str] args) {

    // FIXME: don't hard-wire this.
    auto target_cfg = rec(os = get_os(),
                          arch = session.arch_x86,
                          int_type = common.ty_i32,
                          uint_type = common.ty_u32,
                          float_type = common.ty_f64 );

    auto sess = session.session(target_cfg);
    let option.t[str] input_file = none[str];
    let option.t[str] output_file = none[str];
    let bool do_warn = true;
    let bool shared = false;

    auto i = 1u;
    auto len = _vec.len[str](args);

    // FIXME: a getopt module would be nice.
    while (i < len) {
        auto arg = args.(i);
        if (_str.byte_len(arg) > 0u && arg.(0) == '-' as u8) {
            if (_str.eq(arg, "-nowarn")) {
                do_warn = false;
            } else if (_str.eq(arg, "-shared")) {
                shared = true;
            } else {
                // FIXME: rust could use an elif construct.
                if (_str.eq(arg, "-o")) {
                    if (i+1u < len) {
                        output_file = some(args.(i+1u));
                        i += 1u;
                    } else {
                        usage(sess, args.(0));
                        sess.err("-o requires an argument");
                    }
                } else {
                    if (_str.eq(arg, "-h")) {
                        usage(sess, args.(0));
                    } else {
                        usage(sess, args.(0));
                        sess.err("unrecognized option: " + arg);
                    }
                }
            }
        } else {
            alt (input_file) {
                case (some[str](_)) {
                    usage(sess, args.(0));
                    sess.err("multiple inputs provided");
                }
                case (none[str]) {
                    input_file = some[str](arg);
                }
            }
            // FIXME: dummy node to work around typestate mis-wiring bug.
            i = i;
        }
        i += 1u;
    }

    if (do_warn) {
        warn_wrong_compiler();
    }

    alt (input_file) {
        case (none[str]) {
            usage(sess, args.(0));
            sess.err("no input filename");
        }
        case (some[str](?ifile)) {
            alt (output_file) {
                case (none[str]) {
                    let vec[str] parts = _str.split(ifile, '.' as u8);
                    parts = _vec.pop[str](parts);
                    parts += ".bc";
                    auto ofile = _str.concat(parts);
                    compile_input(sess, ifile, ofile, shared);
                }
                case (some[str](?ofile)) {
                    compile_input(sess, ifile, ofile, shared);
                }
            }
        }
    }
}


// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
