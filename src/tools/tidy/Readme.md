We should add some docs for the tidy tool (probably a README.md in the src/tools/tidy directory then backlink from rustc-dev-guide). Specifically, things like:

    What is tidy
    Why do we have tidy and enforce its checks
    What checks does it perform
    Tidy directives (not to be confused with compiletest directives)
    Interactions between tidy and other tools like compiletest (e.g. revision checks and stuff)




# Tidy

Tidy is a custom tool used for validating source code style and formatting conventions. Though, as you'll see it's also much more than that!

Tidy can be separated into three basic funcationalities:

* Linting and formatting
* Repository Management
* Test helpers

## Linting and formatting

Examples:

* [`alphabetical`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/alphabetical/index.html): Format lists alphabetically
* [`edition`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/edition/index.html): Check Rust edition
* [`error_codes`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/error_codes/index.html): Check to ensure error codes are properly documented and tested
* [`extdeps`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/extdeps/index.html): Check for external package sources
* [`features`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/features/index.html): Check to ensure that unstable features are in order.
* [`filenames`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/filenames/index.html): Check to prevent invalid characters in source.
* [`fluent_alphabetical`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/fluent_alphabetical/index.html) / [`fluent_period`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/fluent_period/index.html) / [`fluent_used`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/fluent_used/index.html): Various checks related to [Fluent](https://github.com/projectfluent) for localization and natural language translation.
* [`pal`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/pal/index.html): Check to enforce rules about platform-specific code in std.
* [`rustdoc_css_themes`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/rustdoc_css_themes/index.html): Tidy check to make sure light and dark themes are synchronized.
* [`rustdoc_gui_tests`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/rustdoc_gui_tests/index.html): Check to ensure that rustdoc GUI tests start with a small description.
* [`rustdoc_json`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/rustdoc_json/index.html): Check to ensure that `FORMAT_VERSION` was correctly updated if `rustdoc-json-types` was updated as well.
* [`rustdoc_templates`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/rustdoc_templates/index.html): Check to ensure that rustdoc templates didn’t forget a `{# #}` to strip extra whitespace characters.
* [`style`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/style/index.html): Check to enforce various stylistic guidelines on the Rust codebase.
* [`target_policy`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/target_policy/index.html): Checks for target tier policy compliance.

* [`target_specific_tests`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/target_specific_tests/index.html): Check to ensure that all target specific tests (those that require a --target flag) also require the pre-requisite LLVM components to run.
* [`tests_placement`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/tests_placement/index.html): Checks that tests are correctly located in `/tests/` and not in `/src/tests`.
* [`unknown_revision`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/unknown_revision/index.html): Checks that test revision names appearing in header directives and error annotations have actually been declared in revisions.
* [`unstable_book`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/unstable_book/index.html): Checks that the [unstable book](https://doc.rust-lang.org/beta/unstable-book/) is synchonized with current unstable features.
* [`x_version`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/x_version/index.html): Checks the current version of the `x` tool and prompts the user to upgrade if out of date.

## Repository Management

* [`bins`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/bins/index.html): Prevent stray binaries from being merged
* [`deps`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/deps/index.html): Check licenses for dependencies
* [`gcc_submodule`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/gcc_submodule/index.html): Check that the src/gcc submodule is the same as the required GCC version of the GCC codegen backend.
* [`triagebot`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/triagebot/index.html): Check to ensure paths mentioned in triagebot.toml exist in the project



## Test Helpers

* [`debug_artifacts`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/debug_artifacts/index.html): Prevent unnecessary debug artifacts while running tests
* [`known_bug`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/known_bug/index.html):  Check to ensure that tests inside `tests/crashes` have a ‘@known-bug’ directive.
* [`mir_opt_tests`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/mir_opt_tests/index.html): Check to ensure that mir opt directories do not have stale files or dashes in file names

* [`tests_revision_unpaired_stdout_stderr`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/tests_revision_unpaired_stdout_stderr/index.html): Checks that there are no unpaired .stderr or .stdout for a test
* [`ui_tests`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/ui_tests/index.html): Check to ensure no stray `.stderr` files in UI test directories. (TODO: this needs a little more digging.)
* [`unit_tests`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/unit_tests/index.html): Check to ensure `#[test]` and `#[bench]` are not used directly inside of the standard library.

## Manual Usage and Extra Checks

[`extra_checks`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/extra_checks/index.html): Optional checks for file types other than Rust source

Example usage:



`./x test tidy --extra-checks=py,cpp,js,spellcheck`

`./x test tidy --extra-checks=py:lint,shell --bless`


All options for `--extra-checks`: `py`, `py:lint`, `py:fmt`, `shell`, `shell:lint`, `cpp`, `cpp:fmt`, `spellcheck`, `js`, `js:lint`, `js:fmt`.

`--bless` when used with tidy applies the formatter and make changes to the source code according to the applied tidy directives.

# Commands

Formatting is checked by the tidy script. It runs automatically when you do `./x test` and can be run in isolation with `./x fmt --check`.

`./x test tidy --extra-checks cpp:fmt --bless`

* --extra-checks=py,shell
* --extra-checks=py:lint
* --extra-checks=py -- foo.py
* --bless performs an actual format, and without bless is just a check (I think).

The main options for --extra-checks are:

    py: Runs checks on Python files.
    py:lint: A more specific check for Python linting.
    shell: Runs checks on shell scripts.
    all: Runs all available extra checks.

    --extra-checks=spellcheck

    let python_lint = extra_check!(Py, Lint);
    let python_fmt = extra_check!(Py, Fmt);
    let shell_lint = extra_check!(Shell, Lint);
    let cpp_fmt = extra_check!(Cpp, Fmt);
    let spellcheck = extra_check!(Spellcheck, None);
    let js_lint = extra_check!(Js, Lint);
    let js_typecheck = extra_check!(Js, Typecheck);

    Full extra checks matrix:

    py:lint, py:fmt, py, shell:lint == shell(?), cpp:fmt == cpp(?), spellcheck, js:lint, js:typecheck

    QUESTION: if you omit the lint/fmt does it run everything? i.e. py runs both lint and fmt?


[Fluent](https://github.com/projectfluent) is for natural language translation.

From `tidy/src/main.rs`:

        // Checks that are done on the cargo workspace.
        // Checks over tests.
        // Checks that only make sense for the compiler.
        // Checks that only make sense for the std libs.
        // Checks that need to be done for both the compiler and std libraries.

# Tidy Directives

        if contents.contains(&format!("// ignore-tidy-{check}"))
            || contents.contains(&format!("# ignore-tidy-{check}"))
            || contents.contains(&format!("/* ignore-tidy-{check} */"))
            || contents.contains(&format!("<!-- ignore-tidy-{check} -->"))
        {

Tidy directives are special comments that tell `Tidy` to operate on a chunk of source code. For example:

```
// tidy-alphabetical-start
fn aaa() {}
fn eee() {}
fn z() {}
// tidy-alphabetical-end
```

Additionally, you can use tidy directives to ignore tidy checks.

The full list of possible exclusions:

* `ignore-tidy-cr`
* `ignore-tidy-undocumented-unsafe`
* `ignore-tidy-tab`
* `ignore-tidy-linelength`
* `ignore-tidy-filelength`
* `ignore-tidy-end-whitespace`
* `ignore-tidy-trailing-newlines`
* `ignore-tidy-leading-newlines`
* `ignore-tidy-copyright`
* `ignore-tidy-dbg`
* `ignore-tidy-odd-backticks`

<!--tidy-alphabetical-start-->
all
okay
umm
banana
applied
zoo
<!--tidy-alphabetical-end-->


`ignore-tidy-*`
`ignore-tidy-{tidy_check}`
`ignore-tidy-pal`
`# ignore-tidy-linelength`
<!--ignore-tidy-todo-->`// TODO` -> Not exactly a directive, but will fail tidy.

    "cr",
    "undocumented-unsafe",
    "tab",
    LINELENGTH_CHECK,

    "filelength",
    "end-whitespace",
    "trailing-newlines",
    "leading-newlines",
    "copyright",
    "dbg",
    "odd-backticks",

# Where is Tidy used?

## CI

## Tests


Questions that should be answered:

What exactly does bless do? Is it a Tidy thing or an `x` thing?
Can you add any tidy check to an ignore-tidy directive?
What happens if you ignore a tidy check that wouldn't run? i.e. `ignore-tidy
