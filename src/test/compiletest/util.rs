import std::option;
import std::generic_os::getenv;
import std::io;
import std::str;

import common::config;

fn make_new_path(path: &istr) -> istr {

    // Windows just uses PATH as the library search path, so we have to
    // maintain the current value while adding our own
    alt getenv(lib_path_env_var()) {
      option::some(curr) {
        #ifmt["%s:%s", path, curr] }
      option::none. { path }
    }
}

#[cfg(target_os = "linux")]
fn lib_path_env_var() -> istr { ~"LD_LIBRARY_PATH" }

#[cfg(target_os = "macos")]
fn lib_path_env_var() -> istr { ~"DYLD_LIBRARY_PATH" }

#[cfg(target_os = "win32")]
fn lib_path_env_var() -> istr { ~"PATH" }

fn logv(config: &config, s: &istr) {
    log s;
    if config.verbose {
        io::stdout().write_line(s);
    }
}
