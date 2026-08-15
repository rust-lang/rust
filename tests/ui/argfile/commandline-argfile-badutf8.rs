// Check to see if we can get parameters from an @argsfile file
//
// Path replacement in .stderr files (i.e. `$DIR`) doesn't handle mixed path
// separators. We have a duplicated version of this test that uses backslash as
// the path separator for the command line arguments that is only run on
// windows.
//
//@ ignore-windows
//@ compile-flags: --cfg cmdline_set @{{src-base}}/argfile/commandline-argfile-badutf8.args

#[cfg(not(cmdline_set))]
compile_error!("cmdline_set not set");

#[cfg(not(unbroken))]
compile_error!("unbroken not set");

fn main() {
}
