## Contributing

### Test and file issues

It would be really useful to have people use rustfmt on their projects and file
issues where it does something you don't expect.

A really useful thing to do that on a crate from the Rust repo. If it does
something unexpected, file an issue; if not, make a PR to the Rust repo with the
reformatted code. We hope to get the whole repo consistently rustfmt'ed and to
replace `make tidy` with rustfmt as a medium-term goal.

### Create test cases

Having a strong test suite for a tool like this is essential. It is very easy
to create regressions. Any tests you can add are very much appreciated.

The tests can be run with `cargo test`. This does a number of things:
* runs the unit tests for a number of internal functions;
* makes sure that rustfmt run on every file in `./tests/source/` is equal to its
  associated file in `./tests/target/`;
* runs idempotence tests on the files in `./tests/target/`. These files should
  not be changed by rustfmt;
* checks that rustfmt's code is not changed by running on itself. This ensures
  that the project bootstraps.

Creating a test is as easy as creating a new file in `./tests/source/` and an
equally named one in `./tests/target/`. If it is only required that rustfmt
leaves a piece of code unformatted, it may suffice to only create a target file.

Whenever there's a discrepancy between the expected output when running tests, a
colourised diff will be printed so that the offending line(s) can quickly be
identified.

Without explicit settings, the tests will be run using rustfmt's default
configuration. It is possible to run a test using non-default settings by
including configuration parameters in comments at the top of the file. For
example: to use 3 spaces per tab, start your test with
`// rustfmt-tab_spaces: 3`. Just remember that the comment is part of the input,
so include in both the source and target files! It is also possible to
explicitly specify the name of the expected output file in the target directory.
Use `// rustfmt-target: filename.rs` for this. Finally, you can use a custom
configuration by using the `rustfmt-config` directive. Rustfmt will then use
that toml file located in `./tests/config/` for its configuration. Including
`// rustfmt-config: small_tabs.toml` will run your test with the configuration
file found at `./tests/config/small_tabs.toml`.

### Hack!

Here are some [good starting issues](https://github.com/nrc/rustfmt/issues?q=is%3Aopen+is%3Aissue+label%3Aeasy).
Note than some of those issues tagged 'easy' are not that easy and might be better
second issues, rather than good first issues to fix.

If you've found areas which need polish and don't have issues, please submit a
PR, don't feel there needs to be an issue.
