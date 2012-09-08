use common::config;
use io::ReaderUtil;

export test_props;
export load_props;
export is_test_ignored;

type test_props = {
    // Lines that should be expected, in order, on standard out
    error_patterns: ~[~str],
    // Extra flags to pass to the compiler
    compile_flags: Option<~str>,
    // If present, the name of a file that this test should match when
    // pretty-printed
    pp_exact: Option<Path>,
    // Modules from aux directory that should be compiled
    aux_builds: ~[~str],
    // Environment settings to use during execution
    exec_env: ~[(~str,~str)]
};

// Load any test directives embedded in the file
fn load_props(testfile: &Path) -> test_props {
    let mut error_patterns = ~[];
    let mut aux_builds = ~[];
    let mut exec_env = ~[];
    let mut compile_flags = option::None;
    let mut pp_exact = option::None;
    for iter_header(testfile) |ln| {
        match parse_error_pattern(ln) {
          option::Some(ep) => vec::push(error_patterns, ep),
          option::None => ()
        };

        if option::is_none(compile_flags) {
            compile_flags = parse_compile_flags(ln);
        }

        if option::is_none(pp_exact) {
            pp_exact = parse_pp_exact(ln, testfile);
        }

        do option::iter(parse_aux_build(ln)) |ab| {
            vec::push(aux_builds, ab);
        }

        do option::iter(parse_exec_env(ln)) |ee| {
            vec::push(exec_env, ee);
        }
    };
    return {
        error_patterns: error_patterns,
        compile_flags: compile_flags,
        pp_exact: pp_exact,
        aux_builds: aux_builds,
        exec_env: exec_env
    };
}

fn is_test_ignored(config: config, testfile: &Path) -> bool {
    let mut found = false;
    for iter_header(testfile) |ln| {
        if parse_name_directive(ln, ~"xfail-test") { return true; }
        if parse_name_directive(ln, xfail_target()) { return true; }
        if config.mode == common::mode_pretty &&
           parse_name_directive(ln, ~"xfail-pretty") { return true; }
    };
    return found;

    fn xfail_target() -> ~str {
        ~"xfail-" + os::sysname()
    }
}

fn iter_header(testfile: &Path, it: fn(~str) -> bool) -> bool {
    let rdr = result::get(io::file_reader(testfile));
    while !rdr.eof() {
        let ln = rdr.read_line();

        // Assume that any directives will be found before the first
        // module or function. This doesn't seem to be an optimization
        // with a warm page cache. Maybe with a cold one.
        if str::starts_with(ln, ~"fn")
            || str::starts_with(ln, ~"mod") {
            return false;
        } else { if !(it(ln)) { return false; } }
    }
    return true;
}

fn parse_error_pattern(line: ~str) -> Option<~str> {
    parse_name_value_directive(line, ~"error-pattern")
}

fn parse_aux_build(line: ~str) -> Option<~str> {
    parse_name_value_directive(line, ~"aux-build")
}

fn parse_compile_flags(line: ~str) -> Option<~str> {
    parse_name_value_directive(line, ~"compile-flags")
}

fn parse_exec_env(line: ~str) -> Option<(~str, ~str)> {
    do parse_name_value_directive(line, ~"exec-env").map |nv| {
        // nv is either FOO or FOO=BAR
        let strs = str::splitn_char(nv, '=', 1u);
        match strs.len() {
          1u => (strs[0], ~""),
          2u => (strs[0], strs[1]),
          n => fail fmt!("Expected 1 or 2 strings, not %u", n)
        }
    }
}

fn parse_pp_exact(line: ~str, testfile: &Path) -> Option<Path> {
    match parse_name_value_directive(line, ~"pp-exact") {
      option::Some(s) => option::Some(Path(s)),
      option::None => {
        if parse_name_directive(line, ~"pp-exact") {
            option::Some(testfile.file_path())
        } else {
            option::None
        }
      }
    }
}

fn parse_name_directive(line: ~str, directive: ~str) -> bool {
    str::contains(line, directive)
}

fn parse_name_value_directive(line: ~str,
                              directive: ~str) -> Option<~str> unsafe {
    let keycolon = directive + ~":";
    match str::find_str(line, keycolon) {
        option::Some(colon) => {
            let value = str::slice(line, colon + str::len(keycolon),
                                   str::len(line));
            debug!("%s: %s", directive,  value);
            option::Some(value)
        }
        option::None => option::None
    }
}
