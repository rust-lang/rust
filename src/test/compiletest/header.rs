import std::option;
import std::istr;
import std::io;
import std::fs;

import common::config;

export test_props;
export load_props;
export is_test_ignored;

type test_props = {
    // Lines that should be expected, in order, on standard out
    error_patterns: [istr],
    // Extra flags to pass to the compiler
    compile_flags: option::t<istr>,
    // If present, the name of a file that this test should match when
    // pretty-printed
    pp_exact: option::t<istr>,
    // FIXME: no-valgrind is a temporary directive until all of run-fail
    // is valgrind-clean
    no_valgrind: bool
};

// Load any test directives embedded in the file
fn load_props(testfile: &istr) -> test_props {
    let error_patterns = [];
    let compile_flags = option::none;
    let pp_exact = option::none;
    let no_valgrind = false;
    for each ln: istr in iter_header(testfile) {
        alt parse_error_pattern(ln) {
          option::some(ep) { error_patterns += [ep]; }
          option::none. { }
        }

        if option::is_none(compile_flags) {
            compile_flags = parse_compile_flags(ln);
        }

        if option::is_none(pp_exact) {
            pp_exact = parse_pp_exact(ln, testfile);
        }

        if no_valgrind == false {
            no_valgrind = parse_name_directive(ln, ~"no-valgrind");
        }
    }
    ret {
        error_patterns: error_patterns,
        compile_flags: compile_flags,
        pp_exact: pp_exact,
        no_valgrind: no_valgrind
    };
}

fn is_test_ignored(config: &config, testfile: &istr) -> bool {
    let found = false;
    for each ln: istr in iter_header(testfile) {
        // FIXME: Can't return or break from iterator
        found = found || parse_name_directive(ln, ~"xfail-test");
        if (config.mode == common::mode_pretty) {
            found = found || parse_name_directive(ln, ~"xfail-pretty");
        }
    }
    ret found;
}

iter iter_header(testfile: &istr) -> istr {
    let rdr = io::file_reader(testfile);
    while !rdr.eof() {
        let ln = rdr.read_line();

        // Assume that any directives will be found before the first
        // module or function. This doesn't seem to be an optimization
        // with a warm page cache. Maybe with a cold one.
        if istr::starts_with(ln, ~"fn")
            || istr::starts_with(ln, ~"mod") {
            break;
        } else { put ln; }
    }
}

fn parse_error_pattern(line: &istr) -> option::t<istr> {
    parse_name_value_directive(line, ~"error-pattern")
}

fn parse_compile_flags(line: &istr) -> option::t<istr> {
    parse_name_value_directive(line, ~"compile-flags")
}

fn parse_pp_exact(line: &istr, testfile: &istr) -> option::t<istr> {
    alt parse_name_value_directive(line, ~"pp-exact") {
      option::some(s) { option::some(s) }
      option::none. {
        if parse_name_directive(line, ~"pp-exact") {
            option::some(fs::basename(testfile))
        } else {
            option::none
        }
      }
    }
}

fn parse_name_directive(line: &istr, directive: &istr) -> bool {
    istr::find(line, directive) >= 0
}

fn parse_name_value_directive(line: &istr,
                              directive: &istr) -> option::t<istr> {
    let keycolon = directive + ~":";
    if istr::find(line, keycolon) >= 0 {
        let colon = istr::find(line, keycolon) as uint;
        let value =
            istr::slice(line, colon + istr::byte_len(keycolon),
                       istr::byte_len(line));
        log #ifmt("%s: %s", directive,
                  value);
        option::some(value)
    } else { option::none }
}
