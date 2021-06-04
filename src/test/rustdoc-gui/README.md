The tests present here are used to test the generated HTML from rustdoc. The
goal is to prevent unsound/unexpected GUI changes.

This is using the [browser-ui-test] framework to do so. It works as follows:

It wraps [puppeteer] to send commands to a web browser in order to navigate and
test what's being currently displayed in the web page.

You can find more information and its documentation in its [repository][browser-ui-test].

[browser-ui-test]: https://github.com/GuillaumeGomez/browser-UI-test/
[puppeteer]: https://pptr.dev/
