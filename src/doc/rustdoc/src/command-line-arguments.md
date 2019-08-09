# Command-line arguments

Here's the list of arguments you can pass to `rustdoc`:

## `-h`/`--help`: help

Using this flag looks like this:

```bash
$ rustdoc -h
$ rustdoc --help
```

This will show `rustdoc`'s built-in help, which largely consists of
a list of possible command-line flags.

Some of `rustdoc`'s flags are unstable; this page only shows stable
options, `--help` will show them all.

## `-V`/`--version`: version information

Using this flag looks like this:

```bash
$ rustdoc -V
$ rustdoc --version
```

This will show `rustdoc`'s version, which will look something
like this:

```text
rustdoc 1.17.0 (56124baa9 2017-04-24)
```

## `-v`/`--verbose`: more verbose output

Using this flag looks like this:

```bash
$ rustdoc -v src/lib.rs
$ rustdoc --verbose src/lib.rs
```

This enables "verbose mode", which means that more information will be written
to standard out. What is written depends on the other flags you've passed in.
For example, with `--version`:

```text
$ rustdoc --verbose --version
rustdoc 1.17.0 (56124baa9 2017-04-24)
binary: rustdoc
commit-hash: hash
commit-date: date
host: host-triple
release: 1.17.0
LLVM version: 3.9
```

## `-r`/`--input-format`: input format

This flag is currently ignored; the idea is that `rustdoc` would support various
input formats, and you could specify them via this flag.

Rustdoc only supports Rust source code and Markdown input formats. If the
file ends in `.md` or `.markdown`, `rustdoc` treats it as a Markdown file.
Otherwise, it assumes that the input file is Rust.


## `-w`/`--output-format`: output format

This flag is currently ignored; the idea is that `rustdoc` would support
various output formats, and you could specify them via this flag.

Rustdoc only supports HTML output, and so this flag is redundant today.

## `-o`/`--output`: output path

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs -o target\\doc
$ rustdoc src/lib.rs --output target\\doc
```

By default, `rustdoc`'s output appears in a directory named `doc` in
the current working directory. With this flag, it will place all output
into the directory you specify.


## `--crate-name`: controlling the name of the crate

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs --crate-name mycrate
```

By default, `rustdoc` assumes that the name of your crate is the same name
as the `.rs` file. `--crate-name` lets you override this assumption with
whatever name you choose.

## `-L`/`--library-path`: where to look for dependencies

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs -L target/debug/deps
$ rustdoc src/lib.rs --library-path target/debug/deps
```

If your crate has dependencies, `rustdoc` needs to know where to find them.
Passing `--library-path` gives `rustdoc` a list of places to look for these
dependencies.

This flag takes any number of directories as its argument, and will use all of
them when searching.


## `--cfg`: passing configuration flags

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs --cfg feature="foo"
```

This flag accepts the same values as `rustc --cfg`, and uses it to configure
compilation. The example above uses `feature`, but any of the `cfg` values
are acceptable.

## `--extern`: specify a dependency's location

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs --extern lazy-static=/path/to/lazy-static
```

Similar to `--library-path`, `--extern` is about specifying the location
of a dependency. `--library-path` provides directories to search in, `--extern`
instead lets you specify exactly which dependency is located where.

## `-C`/`--codegen`: pass codegen options to rustc

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs -C target_feature=+avx
$ rustdoc src/lib.rs --codegen target_feature=+avx

$ rustdoc --test src/lib.rs -C target_feature=+avx
$ rustdoc --test src/lib.rs --codegen target_feature=+avx

$ rustdoc --test README.md -C target_feature=+avx
$ rustdoc --test README.md --codegen target_feature=+avx
```

When rustdoc generates documentation, looks for documentation tests, or executes documentation
tests, it needs to compile some rust code, at least part-way. This flag allows you to tell rustdoc
to provide some extra codegen options to rustc when it runs these compilations. Most of the time,
these options won't affect a regular documentation run, but if something depends on target features
to be enabled, or documentation tests need to use some additional options, this flag allows you to
affect that.

The arguments to this flag are the same as those for the `-C` flag on rustc. Run `rustc -C help` to
get the full list.

## `--passes`: add more rustdoc passes

Using this flag looks like this:

```bash
$ rustdoc --passes list
$ rustdoc src/lib.rs --passes strip-priv-imports
```

An argument of "list" will print a list of possible "rustdoc passes", and other
arguments will be the name of which passes to run in addition to the defaults.

For more details on passes, see [the chapter on them](passes.md).

See also `--no-defaults`.

## `--no-defaults`: don't run default passes

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs --no-defaults
```

By default, `rustdoc` will run several passes over your code. This
removes those defaults, allowing you to use `--passes` to specify
exactly which passes you want.

For more details on passes, see [the chapter on them](passes.md).

See also `--passes`.

## `--test`: run code examples as tests

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs --test
```

This flag will run your code examples as tests. For more, see [the chapter
on documentation tests](documentation-tests.md).

See also `--test-args`.

## `--test-args`: pass options to test runner

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs --test --test-args ignored
```

This flag will pass options to the test runner when running documentation tests.
For more, see [the chapter on documentation tests](documentation-tests.md).

See also `--test`.

## `--target`: generate documentation for the specified target triple

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs --target x86_64-pc-windows-gnu
```

Similar to the `--target` flag for `rustc`, this generates documentation
for a target triple that's different than your host triple.

All of the usual caveats of cross-compiling code apply.

## `--markdown-css`: include more CSS files when rendering markdown

Using this flag looks like this:

```bash
$ rustdoc README.md --markdown-css foo.css
```

When rendering Markdown files, this will create a `<link>` element in the
`<head>` section of the generated HTML. For example, with the invocation above,

```html
<link rel="stylesheet" type="text/css" href="foo.css">
```

will be added.

When rendering Rust files, this flag is ignored.

## `--html-in-header`: include more HTML in <head>

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs --html-in-header header.html
$ rustdoc README.md --html-in-header header.html
```

This flag takes a list of files, and inserts them into the `<head>` section of
the rendered documentation.

## `--html-before-content`: include more HTML before the content

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs --html-before-content extra.html
$ rustdoc README.md --html-before-content extra.html
```

This flag takes a list of files, and inserts them inside the `<body>` tag but
before the other content `rustdoc` would normally produce in the rendered
documentation.

## `--html-after-content`: include more HTML after the content

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs --html-after-content extra.html
$ rustdoc README.md --html-after-content extra.html
```

This flag takes a list of files, and inserts them before the `</body>` tag but
after the other content `rustdoc` would normally produce in the rendered
documentation.


## `--markdown-playground-url`: control the location of the playground

Using this flag looks like this:

```bash
$ rustdoc README.md --markdown-playground-url https://play.rust-lang.org/
```

When rendering a Markdown file, this flag gives the base URL of the Rust
Playground, to use for generating `Run` buttons.


## `--markdown-no-toc`: don't generate a table of contents

Using this flag looks like this:

```bash
$ rustdoc README.md --markdown-no-toc
```

When generating documentation from a Markdown file, by default, `rustdoc` will
generate a table of contents. This flag suppresses that, and no TOC will be
generated.


## `-e`/`--extend-css`: extend rustdoc's CSS

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs -e extra.css
$ rustdoc src/lib.rs --extend-css extra.css
```

With this flag, the contents of the files you pass are included at the bottom
of Rustdoc's `theme.css` file.

While this flag is stable, the contents of `theme.css` are not, so be careful!
Updates may break your theme extensions.

## `--sysroot`: override the system root

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs --sysroot /path/to/sysroot
```

Similar to `rustc --sysroot`, this lets you change the sysroot `rustdoc` uses
when compiling your code.

### `--edition`: control the edition of docs and doctests

Using this flag looks like this:

```bash
$ rustdoc src/lib.rs --edition 2018
$ rustdoc --test src/lib.rs --edition 2018
```

This flag allows rustdoc to treat your rust code as the given edition. It will compile doctests with
the given edition as well. As with `rustc`, the default edition that `rustdoc` will use is `2015`
(the first edition).

