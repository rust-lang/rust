import option;
import str;
import std::io;
import io::reader_util;
import std::fs;

import common::config;

export test_props;
export load_props;
export is_test_ignored;

type test_props = {
    // Lines that should be expected, in order, on standard out
    error_patterns: [str],
    // Extra flags to pass to the compiler
    compile_flags: option<str>,
    // If present, the name of a file that this test should match when
    // pretty-printed
    pp_exact: option<str>
};

// Load any test directives embedded in the file
fn load_props(testfile: str) -> test_props {
    let error_patterns = [];
    let compile_flags = option::none;
    let pp_exact = option::none;
    iter_header(testfile) {|ln|
        alt parse_error_pattern(ln) {
          option::some(ep) { error_patterns += [ep]; }
          option::none { }
        };

        if option::is_none(compile_flags) {
            compile_flags = parse_compile_flags(ln);
        }

        if option::is_none(pp_exact) {
            pp_exact = parse_pp_exact(ln, testfile);
        }
    };
    ret {
        error_patterns: error_patterns,
        compile_flags: compile_flags,
        pp_exact: pp_exact
    };
}

fn is_test_ignored(config: config, testfile: str) -> bool {
    let found = false;
    iter_header(testfile) {|ln|
        // FIXME: Can't return or break from iterator
        found = found || parse_name_directive(ln, "xfail-test");
        found = found || parse_name_directive(ln, xfail_target());
        if (config.mode == common::mode_pretty) {
            found = found || parse_name_directive(ln, "xfail-pretty");
        }
    };
    ret found;

    fn xfail_target() -> str {
        "xfail-" + std::os::target_os()
    }
}

fn iter_header(testfile: str, it: fn(str)) {
    let rdr = result::get(io::file_reader(testfile));
    while !rdr.eof() {
        let ln = rdr.read_line();

        // Assume that any directives will be found before the first
        // module or function. This doesn't seem to be an optimization
        // with a warm page cache. Maybe with a cold one.
        if str::starts_with(ln, "fn")
            || str::starts_with(ln, "mod") {
            break;
        } else { it(ln); }
    }
}

fn parse_error_pattern(line: str) -> option<str> {
    parse_name_value_directive(line, "error-pattern")
}

fn parse_compile_flags(line: str) -> option<str> {
    parse_name_value_directive(line, "compile-flags")
}

fn parse_pp_exact(line: str, testfile: str) -> option<str> {
    alt parse_name_value_directive(line, "pp-exact") {
      option::some(s) { option::some(s) }
      option::none {
        if parse_name_directive(line, "pp-exact") {
            option::some(fs::basename(testfile))
        } else {
            option::none
        }
      }
    }
}

fn parse_name_directive(line: str, directive: str) -> bool {
    str::find(line, directive) >= 0
}

fn parse_name_value_directive(line: str,
                              directive: str) -> option<str> {
    let keycolon = directive + ":";
    if str::find(line, keycolon) >= 0 {
        let colon = str::find(line, keycolon) as uint;
        let value =
            str::slice(line, colon + str::byte_len(keycolon),
                       str::byte_len(line));
        #debug("%s: %s", directive,  value);
        option::some(value)
    } else { option::none }
}
