import std::option;
import std::str;
import std::io;

import common::config;

export test_props;
export load_props;
export is_test_ignored;

type test_props = {error_patterns: str[], compile_flags: option::t[str]};

// Load any test directives embedded in the file
fn load_props(testfile: &str) -> test_props {
    let error_patterns = ~[];
    let compile_flags = option::none;
    for each ln: str  in iter_header(testfile) {
        alt parse_error_pattern(ln) {
          option::some(ep) { error_patterns += ~[ep]; }
          option::none. { }
        }


        if option::is_none(compile_flags) {
            compile_flags = parse_compile_flags(ln);
        }
    }
    ret {error_patterns: error_patterns, compile_flags: compile_flags};
}

fn is_test_ignored(config: &config, testfile: &str) -> bool {
    let found = false;
    for each ln: str  in iter_header(testfile) {
        // FIXME: Can't return or break from iterator
        found = found
            || parse_name_directive(ln, "xfail-" + config.stage_id);
        if (config.mode == common::mode_pretty) {
            found = found
                || parse_name_directive(ln, "xfail-pretty");
        }
    }
    ret found;
}

iter iter_header(testfile: &str) -> str {
    let rdr = io::file_reader(testfile);
    while !rdr.eof() {
        let ln = rdr.read_line();

        // Assume that any directives will be found before the first
        // module or function. This doesn't seem to be an optimization
        // with a warm page cache. Maybe with a cold one.
        if str::starts_with(ln, "fn") || str::starts_with(ln, "mod") {
            break;
        } else { put ln; }
    }
}

fn parse_error_pattern(line: &str) -> option::t[str] {
    parse_name_value_directive(line, "error-pattern")
}

fn parse_compile_flags(line: &str) -> option::t[str] {
    parse_name_value_directive(line, "compile-flags")
}

fn parse_name_directive(line: &str, directive: &str) -> bool {
    str::find(line, directive) >= 0
}

fn parse_name_value_directive(line: &str,
                              directive: &str) -> option::t[str] {
    let keycolon = directive + ":";
    if str::find(line, keycolon) >= 0 {
        let colon = str::find(line, keycolon) as uint;
        let value =
            str::slice(line, colon + str::byte_len(keycolon),
                       str::byte_len(line));
        log #fmt("%s: %s", directive, value);
        option::some(value)
    } else { option::none }
}
