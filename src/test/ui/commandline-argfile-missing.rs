// Check to see if we can get parameters from an @argsfile file
//
// build-fail
// normalize-stderr-test: "Argument \d+" -> "Argument $$N"
// normalize-stderr-test: "os error \d+" -> "os error $$ERR"
// compile-flags: --cfg cmdline_set @{{src-base}}/commandline-argfile-missing.args

#[cfg(not(cmdline_set))]
compile_error!("cmdline_set not set");

#[cfg(not(unbroken))]
compile_error!("unbroken not set");

fn main() {
}
