# Writing rustc-dev-guide documentation

Contributions to the [rustc-dev-guide] are always welcome, and can be made directly at
[the rust-lang/rustc-dev-guide repo][rdgrepo].
The issue tracker in that repo is also a great way to find things that need doing.
There are issues for beginners and advanced compiler devs alike!

Just a few things to keep in mind:

[rustc API docs]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle

- When writing about a particular part of the compiler's code, we
  recommend that you link to the relevant parts of the [rustc API docs].

- Use sentence case for chapter and section titles.

- Use dashes (`-`) to separate words in file names.

- Links within the guide should use `.md` relative links, not `.html` links.
  CI will enforce this.

- Please try to avoid overly long lines and use semantic line breaks (where you break the line after each sentence).
  This makes it easier to review diffs, since they avoid reflowing other unrelated prose.
  There is no strict limit on line lengths;
  let the sentence or part of the sentence flow to its proper end on the same line.

  You can use a tool in ci/sembr to help with this.
  Its help output can be seen with this command:

  ```console
  cargo run --manifest-path ci/sembr/Cargo.toml -- --help
  ```

- When contributing text to the guide, please contextualize the information with some time period
  and/or a reason so that the reader knows how much to trust the information.
  Aim to provide a reasonable amount of context, and consider including:

  - A reason for why the text may be out of date other than "change",
    as change is a constant across the project.

  - The date the comment was added, e.g. instead of writing _"Currently, ..."_
    or _"As of now, ..."_, consider adding the date, in one of the following formats:
    - Jan 2021
    - January 2021
    - jan 2021
    - january 2021

    There is a CI action (in `.github/workflows/date-check.yml`)
    that generates a monthly report showing those that are over 6 months old
    ([example](https://github.com/rust-lang/rustc-dev-guide/issues/2052)).

    For the action to pick the date, add a special annotation before specifying the date:

    ```md
    <!-- date-check --> Jul 2026
    ```

    Example:

    ```md
    As of <!-- date-check --> Jul 2026, the foo did the bar.
    ```

    For cases where the date should not be part of the visible rendered output,
    use the following instead:

    ```md
    <!-- date-check: Jul 2026 -->
    ```

  - A link to a relevant WG, tracking issue, `rustc` rustdoc page, or similar, that may provide
    further explanation for the change process or a way to verify that the information is not
    outdated.

## ⚠️ Note: Where to contribute `rustc-dev-guide` changes

For detailed information about where to contribute rustc-dev-guide changes and the benefits of doing so,
see [the rustc-dev-guide team documentation].

[rustc-dev-guide]: https://rustc-dev-guide.rust-lang.org/
[rdgrepo]: https://github.com/rust-lang/rustc-dev-guide
[the rustc-dev-guide team documentation]: https://forge.rust-lang.org/rustc-dev-guide/index.html#where-to-contribute-rustc-dev-guide-changes
