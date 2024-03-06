// Check to see if we can get parameters from an @argsfile file
//
//@ build-pass
//@ compile-flags: @{{src-base}}/shell-argfiles/shell-argfiles-via-argfile.args @shell:{{src-base}}/shell-argfiles/shell-argfiles-via-argfile-shell.args

#[cfg(not(shell_args_set))]
compile_error!("shell_args_set not set");

fn main() {
}
