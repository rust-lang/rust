This is serves as a collection of crashes so that accidental ICE fixes are tracked.
This was formally done at https://github.com/rust-lang/glacier but doing it inside
the rustc testsuite is more convenient.

It is imperative that a test in the suite causes an internal compiler error/panic
or makes rustc crash in some other way.
A test will "pass" if rustc exits with something other than 1 or 0.

When adding crashes from https://github.com/rust-lang/rust/issues, the
issue number should be noted in the file name (12345.rs should suffice)
and also inside the file via `//@ known-bug #4321` if possible.

If you happen to fix one of the crashes, please move it to a fitting
subdirectory in `tests/ui` and give it a meaningful name.
Also please add a doc comment at the top of the file explaining why
this test exists. :)
Adding
Fixes #NNNNN
Fixes #MMMMM
to the description of your pull request will ensure the
corresponding tickets will be closed automatically upon merge.
The ticket ids can be found in the file name or the `known-bug` annotation
inside the testfile.

Please do not re-report any crashes that you find here!
