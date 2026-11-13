HEADS UP! https://example.com MUST SHOW UP IN THE STDERR FILE!

Normally, a line with errors on it will also have a comment
marking it up as something that needs to generate an error.

The test harness doesn't gather hot comments from this file.
Rustdoc will generate an error for the line, and the `.stderr`
snapshot includes this error, but Compiletest doesn't see it.

If the stderr file changes, make sure the warning points at the URL!
