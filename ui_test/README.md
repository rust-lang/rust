A smaller version of compiletest-rs

## Magic behavior

* Tests are run in order of their filenames (files first, then recursing into folders).
  So if you have any slow tests, prepend them with a small integral number to make them get run first, taking advantage of parallelism as much as possible (instead of waiting for the slow tests at the end).

## Supported magic comment annotations

If your test tests for failure, you need to add a `//~` annotation where the error is happening
to make sure that the test will always keep failing with a specific message at the annotated line.

`//~ ERROR: XXX` make sure the stderr output contains `XXX` for an error in the line where this comment is written

* Also supports `HELP`, `WARN` or `NOTE` for different kind of message
    * if one of those levels is specified explicitly, *all* diagnostics of this level or higher need an annotation. If you want to avoid this, just leave out the all caps level note entirely.
* If the all caps note is left out, a message of any level is matched. Leaving it out is not allowed for `ERROR` levels.
* This checks the output *before* normalization, so you can check things that get normalized away, but need to
    be careful not to accidentally have a pattern that differs between platforms.
* if `XXX` is of the form `/XXX/` it is treated as a regex instead of a substring and will succeed if the regex matches.

In order to change how a single test is tested, you can add various `//@` comments to the test.
Any other comments will be ignored, and all `//@` comments must be formatted precisely as
their command specifies, or the test will fail without even being run.

* `//@ignore-XXX` avoids running the test on targets whose triple contains `XXX`
    * `XXX` can also be one of `64bit`, `32bit` or `16bit`
    * `XXX` can also be `on-host`, which will only run the test during cross compilation testing.
* `//@only-XXX` avoids running the test on targets whose triple **does not** contain `XXX`
    * `XXX` can also be one of `64bit`, `32bit` or `16bit`
    * `XXX` can also be `on-host`, which will not run the test during cross compilation testing
* `//@stderr-per-bitwidth` produces one stderr file per bitwidth, as they may differ significantly sometimes
* `//@error-pattern: XXX` make sure the stderr output contains `XXX`
* `//@revisions: XXX YYY` runs the test once for each space separated name in the list
    * emits one stderr file per revision
    * `//~` comments can be restricted to specific revisions by adding the revision name before the `~` in square brackets: `//[XXX]~`
* `//@compile-flags: XXX` appends `XXX` to the command line arguments passed to the rustc driver
    * you can specify this multiple times, and all the flags will accumulate
* `//@rustc-env: XXX=YYY` sets the env var `XXX` to `YYY` for the rustc driver execution.
    * for Miri these env vars are used during compilation via rustc and during the emulation of the program
    * you can specify this multiple times, accumulating all the env vars
* `//@normalize-stderr-test: "REGEX" -> "REPLACEMENT"` replaces all matches of `REGEX` in the stderr with `REPLACEMENT`. The replacement may specify `$1` and similar backreferences to paste captures.
    * you can specify multiple such commands, there is no need to create a single regex that handles multiple replacements that you want to perform.
* `//@require-annotations-for-level: LEVEL` can be used to change the level of diagnostics that require a corresponding annotation.
    * this is only useful if there are any annotations like `HELP`, `WARN` or `NOTE`, as these would automatically require annotations for all other diagnostics of the same or higher level.

## Significant differences to compiletest-rs

* `ignore-*` and `only-*` opereate solely on the triple, instead of supporting things like `macos`
* only `//~` comments can be individualized per revision
