# Guide to the UI Tests

The UI tests are intended to capture the compiler's complete output,
so that we can test all aspects of the presentation. They work by
compiling a file (e.g., `hello_world/main.rs`), capturing the output,
and then applying some normalization (see below). This normalized
result is then compared against reference files named
`hello_world/main.stderr` and `hello_world/main.stdout`. If either of
those files doesn't exist, the output must be empty. If the test run
fails, we will print out the current output, but it is also saved in
`build/<target-triple>/test/ui/hello_world/main.stdout` (this path is
printed as part of the test failure mesage), so you can run `diff` and
so forth.

# Editing and updating the reference files

If you have changed the compiler's output intentionally, or you are
making a new test, you can use the script `update-references.sh` to
update the references. When you run the test framework, it will report
various errors: in those errors is a command you can use to run the
`update-references.sh` script, which will then copy over the files
from the build directory and use them as the new reference. You can
also just run `update-all-references.sh`. In both cases, you can run
the script with `--help` to get a help message.

# Normalization

The normalization applied is aimed at filenames:

- the test directory is replaced with `$DIR`
- all backslashes (\) are converted to forward slashes (/) (for windows)
