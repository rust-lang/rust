The tests present here are used to test the generated HTML from rustdoc. The
goal is to prevent unsound/unexpected GUI changes.

This is using the [browser-ui-test] framework to do so. It works as follows:

It wraps [puppeteer] to send commands to a web browser in order to navigate and
test what's being currently displayed in the web page.

You can find more information and its documentation in its [repository][browser-ui-test].

If you need to have more information on the tests run, you can use `--test-args`:

```bash
$ ./x.py test src/test/rustdoc-gui --stage 1 --jobs 8 --test-args --debug
```

There are three options supported:

 * `--debug`: allows to see puppeteer commands.
 * `--no-headless`: disable headless mode so you can see what's going on.
 * `--show-text`: by default, text isn't rendered because of issues with fonts, it enables it back.

[browser-ui-test]: https://github.com/GuillaumeGomez/browser-UI-test/
[puppeteer]: https://pptr.dev/
