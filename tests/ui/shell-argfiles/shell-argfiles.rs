// Check to see if we can get parameters from an @argsfile file
//
//@ build-pass
//@ compile-flags: --cfg cmdline_set -Z shell-argfiles @shell:{{src-base}}/shell-argfiles/shell-argfiles.args

#[cfg(not(cmdline_set))]
compile_error!("cmdline_set not set");

#[cfg(not(unquoted_set))]
compile_error!("unquoted_set not set");

#[cfg(not(single_quoted_set))]
compile_error!("single_quoted_set not set");

#[cfg(not(double_quoted_set))]
compile_error!("double_quoted_set not set");

fn main() {
}
