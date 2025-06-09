# The `rustdoc-gui` test suite

> **FIXME**: This section is a stub. Please help us flesh it out!

This page is about the test suite named `rustdoc-gui` used to test the "GUI" of `rustdoc` (i.e., the HTML/JS/CSS as rendered in a browser).
For other rustdoc-specific test suites, see [Rustdoc test suites].

These use a NodeJS-based tool called [`browser-UI-test`] that uses [puppeteer] to run tests in a headless browser and check rendering and interactivity. For information on how to write this form of test, see [`tests/rustdoc-gui/README.md`][rustdoc-gui-readme] as well as [the description of the `.goml` format][goml-script]

[Rustdoc test suites]: ../tests/compiletest.md#rustdoc-test-suites
[`browser-UI-test`]: https://github.com/GuillaumeGomez/browser-UI-test/
[puppeteer]: https://pptr.dev/
[rustdoc-gui-readme]: https://github.com/rust-lang/rust/blob/master/tests/rustdoc-gui/README.md
[goml-script]: https://github.com/GuillaumeGomez/browser-UI-test/blob/master/goml-script.md
