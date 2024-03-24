# target-docs

This tool generates target documentation for all targets in the rustc book.

To achieve this, it uses a list of input markdown files provided in `src/doc/rustc/target_infos`. These files follow a strict format.
Every file covers a glob pattern of targets according to its file name.

For every rustc target, we iterate through all the target infos and find matching globs.
When a glob matches, it extracts the h2 markdown sections and saves them for the target.

In the end, a page is generated for every target using these sections.
Sections that are not provided are stubbed out. Currently, the sections are

- Overview
- Requirements
- Testing
- Building the target
- Cross compilation
- Building Rust programs

In addition to the markdown sections, we also have extra data about the targets.
This is achieved through YAML frontmatter.

The frontmatter follows the following format:

```yaml
maintainers: ["@someone"]
footnotes:
  i686-pc-windows-gnu:
    - "x86_32-floats-return-ABI"
    - "windows-support"
```

The top level keys are:

- `maintainers` (optional): list of strings
- `footnotes` (optional): for every *specific* target a list of footnotes. The footnotes have to be defined manually below the correct table in platform-support.
