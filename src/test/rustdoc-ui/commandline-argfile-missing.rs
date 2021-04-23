// Check to see if we can get parameters from an @argsfile file
//
// normalize-stderr-test: "os error \d+" -> "os error $$ERR"
// normalize-stderr-test: "commandline-argfile-missing.args:[^(]*" -> "commandline-argfile-missing.args: $$FILE_MISSING "
// compile-flags: --cfg cmdline_set @{{src-base}}/commandline-argfile-missing.args

#[cfg(not(cmdline_set))]
compile_error!("cmdline_set not set");

#[cfg(not(unbroken))]
compile_error!("unbroken not set");

fn main() {
}
