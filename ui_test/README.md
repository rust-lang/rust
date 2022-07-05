A smaller version of compiletest-rs

## Magic behavior

* Tests are run in order of their filenames (files first, then recursing into folders).
  So if you have any slow tests, prepend them with a small integral number to make them get run first, taking advantage of parallelism as much as possible (instead of waiting for the slow tests at the end).

## Supported magic comment annotations

* `// ignore-XXX` avoids running the test on targets whose triple contains `XXX`
    * `XXX` can also be one of `64bit`, `32bit` or `16bit`
* `// only-XXX` avoids running the test on targets whose triple **does not** contain `XXX`
    * `XXX` can also be one of `64bit`, `32bit` or `16bit`
* `// stderr-per-bitwidth` produces one stderr file per bitwidth, as they may differ significantly sometimes
* `//@error-pattern: XXX` make sure the stderr output contains `XXX`
* `//~ ERROR: XXX` make sure the stderr output contains `XXX` for an error in the line where this comment is written
    * Also supports `HELP`, `WARN` or `NOTE` for different kind of message
        * if one of those levels is specified explicitly, *all* diagnostics of this level or higher need an annotation. If you want to avoid this, just leave out the all caps level note entirely.
    * If the all caps note is left out, a message of any level is matched. Leaving it out is not allowed for `ERROR` levels.
    * This checks the output *before* normalization, so you can check things that get normalized away, but need to
      be careful not to accidentally have a pattern that differs between platforms.
* `// revisions: XXX YYY` runs the test once for each space separated name in the list
    * emits one stderr file per revision
    * `//~` comments can be restricted to specific revisions by adding the revision name before the `~` in square brackets: `//[XXX]~`
* `// compile-flags: XXX` appends `XXX` to the command line arguments passed to the rustc driver
* `// rustc-env: XXX=YYY` sets the env var `XXX` to `YYY` for the rustc driver execution.
    * for Miri these env vars are used during compilation via rustc and during the emulation of the program
* `// normalize-stderr-test: "REGEX" -> "REPLACEMENT"` replaces all matches of `REGEX` in the stderr with `REPLACEMENT`. The replacement may specify `$1` and similar backreferences to paste captures.

## Significant differences to compiletest-rs

* `ignore-*` and `only-*` opereate solely on the triple, instead of supporting things like `macos`
* only `//~` comments can be individualized per revision
