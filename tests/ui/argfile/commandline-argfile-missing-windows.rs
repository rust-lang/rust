// Check to see if we can get parameters from an @argsfile file
//
// Path replacement in .stderr files (i.e. `$DIR`) doesn't handle mixed path
// separators. This test uses backslash as the path separator for the command
// line arguments and is only run on windows.
//
//@ only-windows
//@ normalize-stderr-test: "os error \d+" -> "os error $$ERR"
//@ normalize-stderr-test: "commandline-argfile-missing.args:[^(]*" -> "commandline-argfile-missing.args: $$FILE_MISSING "
//@ compile-flags: --cfg cmdline_set @{{src-base}}\argfile\commandline-argfile-missing.args

#[cfg(not(cmdline_set))]
compile_error!("cmdline_set not set");

#[cfg(not(unbroken))]
compile_error!("unbroken not set");

fn main() {
}
