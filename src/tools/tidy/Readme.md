# Tidy
Tidy is the Rust project's custom internal linter and a crucial part of our testing and continuous integration (CI) infrastructure. It is designed to enforce a consistent style and formatting across the entire codebase, but its role extends beyond simple linting. Tidy also helps with infrastructure, policy, and documentation, ensuring the project remains organized, functional, and... tidy.

This document will cover how to use tidy, the specific checks tidy performs, and using tidy directives to manage its behavior. By understanding and utilizing tidy, you can help us maintain the high standards of the Rust project.
## Tidy Checks
### Style and Code Quality
These lints focus on enforcing consistent formatting, style, and general code health.
* [`alphabetical`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/alphabetical/index.html): Checks that lists are sorted alphabetically
* [`style`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/style/index.html): Check to enforce various stylistic guidelines on the Rust codebase.
* [`filenames`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/filenames/index.html): Check to prevent invalid characters in file names.
* [`pal`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/pal/index.html): Check to enforce rules about platform-specific code in std.
* [`target_policy`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/target_policy/index.html): Check for target tier policy compliance.
* [`error_codes`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/error_codes/index.html): Check to ensure error codes are properly documented and tested.

### Infrastructure
These checks focus on the integrity of the project's dependencies, internal tools, and documentation.
* [`bins`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/bins/index.html): Prevent stray binaries from being checked into the source tree.
* [`fluent_alphabetical`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/fluent_alphabetical/index.html) / [`fluent_period`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/fluent_period/index.html) / [`fluent_used`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/fluent_used/index.html): Various checks related to [Fluent](https://github.com/projectfluent) for localization and natural language translation.
* [`deps`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/deps/index.html) / [`extdeps`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/extdeps/index.html): Check for valid licenses and sources for external dependencies.
* [`gcc_submodule`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/gcc_submodule/index.html): Check that the `src/gcc` submodule version is valid.
* [`triagebot`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/triagebot/index.html): Check to ensure paths mentioned in `triagebot.toml` exist in the project.
* [`x_version`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/x_version/index.html): Validates the current version of the `x` tool.

* [`edition`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/edition/index.html) / [`features`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/features/index.html): Check for a valid Rust edition and proper ordering of unstable features.
* [`rustdoc_css_themes`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/rustdoc_css_themes/index.html) / [`rustdoc_templates`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/rustdoc_templates/index.html): Verify that Rust documentation templates and themes are correct.
* [`unstable_book`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/unstable_book/index.html): Synchronizes the unstable book with unstable features.
### Testing
These checks ensure that tests are correctly structured, cleaned up, and free of common errors.
* [`tests_placement`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/tests_placement/index.html) / [`unit_tests`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/unit_tests/index.html): Verify that tests are located in the correct directories and are not using improper attributes.
* [`known_bug`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/known_bug/index.html) / [`unknown_revision`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/unknown_revision/index.html): Ensure that test directives and annotations are used correctly.
* [`debug_artifacts`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/debug_artifacts/index.html) / [`mir_opt_tests`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/mir_opt_tests/index.html): Prevent unnecessary artifacts and stale files in test directories.
* [`tests_revision_unpaired_stdout_stderr`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/tests_revision_unpaired_stdout_stderr/index.html) / [`ui_tests`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/ui_tests/index.html): Check for unpaired or stray test output files.
* [`target_specific_tests`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/target_specific_tests/index.html): Check to ensure that all target specific tests (those that require a --target flag) also require the pre-requisite LLVM components to run.
* [`rustdoc_gui_tests`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/rustdoc_gui_tests/index.html): Checks that rustdoc gui tests start with a small description
* [`rustdoc_json`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/rustdoc_json/index.html): Verify that `FORMAT_VERSION` is in sync with `rust-json-types`.
## Using Tidy

Tidy is used in a number of different ways.
* Every time `./x test` is used tidy will run automatically.

* On every pull request, tidy will run automatically during CI checks.
* Optionally, with the use of git-hooks, tidy can run locally on every push. This can be setup with `./x setup`. See the [rustc-dev-guide](https://rustc-dev-guide.rust-lang.org/building/suggested.html#installing-a-pre-push-hook) for more information.

You can run tidy manually with:

`./x test tidy`

To first run the relevant formatter and then run tidy you can add `--bless`.

`./x test tidy --bless`
### Extra Checks
[`extra_checks`](https://doc.rust-lang.org/nightly/nightly-rustc/tidy/extra_checks/index.html) are optional checks primarily focused on other file types and programming languages.

Example usage:

`./x test tidy --extra-checks=py,cpp,js,spellcheck`

All options for `--extra-checks`:
* `cpp`, `cpp:fmt`
* `py`, `py:lint`, `py:fmt`
* `js`, `js:lint`, `js:fmt`, `js:typecheck`
* `shell`, `shell:lint`
* `spellcheck`

Default values for tidy's `extra-checks` can be set in `bootstrap.toml`. For example, `build.tidy-extra-checks = "js,py"`.

Any argument without a suffix (eg. `py` or `js`) will include all possible checks. For example, `--extra-checks=js` is the same as `extra-checks=js:lint,js:fmt,js:typecheck`.

Any argument can be prefixed with `auto:` to only run if relevant files are modified (eg. `--extra-checks=auto:py`).

A specific configuration file or folder can be passed to tidy after a double dash (`--extra-checks=py -- foo.py`)

## Tidy Directives

Tidy directives are special comments that help tidy operate.

Tidy directives can be used in the following types of comments:
* `// `
* `# `
* `/* {...} */`
* `<!-- {...} -->`

You might find yourself needing to ignore a specific tidy style check and can do so with:
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
* `ignore-tidy-todo`

Some checks, like `alphabetical`, require a tidy directive to use:
```
// tidy-alphabetical-start
fn aaa() {}
fn eee() {}
fn z() {}
// tidy-alphabetical-end
```
<!--ignore-tidy-todo-->While not exactly a tidy directive, // TODO will fail tidy and make sure you can't merge a PR with unfinished work.

### Test Specific Directives

`target-specific-tests` can be ignored with `// ignore-tidy-target-specific-tests`

Tidy's `unknown_revision` check can be suppressed by adding the revision name to `//@ unused-revision-names:{revision}` or with `//@ unused-revision-names:*`.
