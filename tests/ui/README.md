# UI Tests

This folder contains `rustc`'s
[UI tests](https://rustc-dev-guide.rust-lang.org/tests/ui.html).

## Test Directives (Headers)

Typically, a UI test will have some test directives / headers which are
special comments that tell compiletest how to build and intepret a test.

As part of an on-going effort to rewrite compiletest
(see <https://github.com/rust-lang/compiler-team/issues/536>), a major
change proposal to change legacy compiletest-style headers `// <directive>`
to [`ui_test`](https://github.com/oli-obk/ui_test)-style headers
`//@ <directive>` was accepted (see
<https://github.com/rust-lang/compiler-team/issues/512>.

An example directive is `ignore-test`. In legacy compiletest style, the header
would be written as

```rs
// ignore-test
```

but in `ui_test` style, the header would be written as

```rs
//@ ignore-test
```

compiletest is changed to accept only `//@` directives for UI tests
(currently), and will reject and report an error if it encounters any
comments `// <content>` that may be parsed as an legacy compiletest-style
test header. To fix this, you should migrate to the `ui_test`-style header
`//@ <content>`.
