A smaller version of compiletest-rs

## Supported magic comment annotations

Note that the space after `//`, when it is present, is *not* optional -- it must be exactly one.

* `// ignore-XXX` avoids running the test on targets whose triple contains `XXX`
    * `XXX` can also be one of `64bit`, `32bit` or `16bit`
* `// only-XXX` avoids running the test on targets whose triple **does not** contain `XXX`
    * `XXX` can also be one of `64bit`, `32bit` or `16bit`
* `// stderr-per-bitwidth` produces one stderr file per bitwidth, as they may differ significantly sometimes
* `// error-pattern: XXX` make sure the stderr output contains `XXX`
* `//~ ERROR: XXX` make sure the stderr output contains `XXX` for an error in the line where this comment is written
    * NOTE: it is not checked at present that it is actually in the line where the error occurred, or that it is truly an ERROR/WARNING/HELP/NOTE, but you should treat it as such until that becomes true.
    * Also supports `HELP` or `WARN` for different kind of message
    * if the all caps note is left out, any message is matched
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
