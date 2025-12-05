The tests present here are used to test the generated HTML from rustdoc. The
goal is to prevent unsound/unexpected GUI changes.

This is using the [browser-ui-test] framework to do so. It works as follows:

It wraps [puppeteer] to send commands to a web browser in order to navigate and
test what's being currently displayed in the web page.

You can find more information and its documentation in its [repository][browser-ui-test].

If you need to have more information on the tests run, you can use `--test-args`:

```bash
$ ./x.py test tests/rustdoc-gui --stage 1 --test-args --debug
```

If you don't want to run in headless mode (helpful to debug sometimes), you can use
`--no-headless`:

```bash
$ ./x.py test tests/rustdoc-gui --stage 1 --test-args --no-headless
```

To see the supported options, use `--help`.

[browser-ui-test]: https://github.com/GuillaumeGomez/browser-UI-test/
[puppeteer]: https://pptr.dev/
