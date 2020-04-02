This document is an index of features that the rust-analyzer language server
provides. Shortcuts are for the default VS Code layout. If there's no shortcut,
you can use <kbd>Ctrl+Shift+P</kbd> to search for the corresponding action.

### Workspace Symbol <kbd>ctrl+t</kbd>

Uses fuzzy-search to find types, modules and functions by name across your
project and dependencies. This is **the** most useful feature, which improves code
navigation tremendously. It mostly works on top of the built-in LSP
functionality, however `#` and `*` symbols can be used to narrow down the
search. Specifically,

- `Foo` searches for `Foo` type in the current workspace
- `foo#` searches for `foo` function in the current workspace
- `Foo*` searches for `Foo` type among dependencies, including `stdlib`
- `foo#*` searches for `foo` function among dependencies

That is, `#` switches from "types" to all symbols, `*` switches from the current
workspace to dependencies.

### Document Symbol <kbd>ctrl+shift+o</kbd>

Provides a tree of the symbols defined in the file. Can be used to

* fuzzy search symbol in a file (super useful)
* draw breadcrumbs to describe the context around the cursor
* draw outline of the file

### On Typing Assists

Some features trigger on typing certain characters:

- typing `let =` tries to smartly add `;` if `=` is followed by an existing expression
- Enter inside comments automatically inserts `///`
- typing `.` in a chain method call auto-indents

### Extend Selection

Extends the current selection to the encompassing syntactic construct
(expression, statement, item, module, etc). It works with multiple cursors. This
is a relatively new feature of LSP:
https://github.com/Microsoft/language-server-protocol/issues/613, check your
editor's LSP library to see if this feature is supported.

### Go to Definition

Navigates to the definition of an identifier.

### Go to Implementation

Navigates to the impl block of structs, enums or traits. Also implemented as a code lens.

### Go to Type Defintion

Navigates to the type of an identifier.

### Commands <kbd>ctrl+shift+p</kbd>

#### Run

Shows a popup suggesting to run a test/benchmark/binary **at the current cursor
location**. Super useful for repeatedly running just a single test. Do bind this
to a shortcut!

#### Parent Module

Navigates to the parent module of the current module.

#### Matching Brace

If the cursor is on any brace (`<>(){}[]`) which is a part of a brace-pair,
moves cursor to the matching brace. It uses the actual parser to determine
braces, so it won't confuse generics with comparisons.

#### Join Lines

Join selected lines into one, smartly fixing up whitespace and trailing commas.

#### Show Syntax Tree

Shows the parse tree of the current file. It exists mostly for debugging
rust-analyzer itself.

#### Expand Macro Recursively

Shows the full macro expansion of the macro at current cursor.

#### Status

Shows internal statistic about memory usage of rust-analyzer.

#### Show RA Version

Show current rust-analyzer version.

#### Run Garbage Collection

Manually triggers GC.

#### Start Cargo Watch

Start `cargo watch` for live error highlighting. Will prompt to install if it's not already installed.

#### Stop Cargo Watch

Stop `cargo watch`.

#### Structural Seach and Replace

Search and replace with named wildcards that will match any expression.
The syntax for a structural search replace command is `<search_pattern> ==>> <replace_pattern>`. A `$<name>:expr` placeholder in the search pattern will match any expression and `$<name>` will reference it in the replacement. Available via the command `rust-analyzer.ssr`.

```rust
// Using structural search replace command [foo($a:expr, $b:expr) ==>> ($a).foo($b)]

// BEFORE
String::from(foo(y + 5, z))

// AFTER
String::from((y + 5).foo(z))
```

### Assists (Code Actions)

Assists, or code actions, are small local refactorings, available in a particular context.
They are usually triggered by a shortcut or by clicking a light bulb icon in the editor.

See [assists.md](./assists.md) for the list of available assists.

### Magic Completions

In addition to usual reference completion, rust-analyzer provides some ✨magic✨
completions as well:

Keywords like `if`, `else` `while`, `loop` are completed with braces, and cursor
is placed at the appropriate position. Even though `if` is easy to type, you
still want to complete it, to get ` { }` for free! `return` is inserted with a
space or `;` depending on the return type of the function.

When completing a function call, `()` are automatically inserted. If a function
takes arguments, the cursor is positioned inside the parenthesis.

There are postifx completions, which can be triggerd by typing something like
`foo().if`. The word after `.` determines postifx completion. Possible variants are:

- `expr.if` -> `if expr {}`
- `expr.match` -> `match expr {}`
- `expr.while` -> `while expr {}`
- `expr.ref` -> `&expr`
- `expr.refm` -> `&mut expr`
- `expr.not` -> `!expr`
- `expr.dbg` -> `dbg!(expr)`

There also snippet completions:

#### Inside Expressions

- `pd` -> `println!("{:?}")`
- `ppd` -> `println!("{:#?}")`

#### Inside Modules

- `tfn` -> `#[test] fn f(){}`

### Code Highlighting

Experimental feature to let rust-analyzer highlight Rust code instead of using the
default highlighter.

#### Rainbow Highlighting

Experimental feature that, given code highlighting using rust-analyzer is
active, will pick unique colors for identifiers.

### Code hints

Rust-analyzer has two types of hints to show the information about the code:

* hover hints, appearing on hover on any element.

These contain extended information on the hovered language item.

* inlay hints, shown near the element hinted directly in the editor.

Two types of inlay hints are displayed currently:

* type hints, displaying the minimal information on the type of the expression (if the information is available)
* method chaining hints, type information for multi-line method chains
* parameter name hints, displaying the names of the parameters in the corresponding methods

#### VS Code

In VS Code, the following settings can be used to configure the inlay hints:

* `rust-analyzer.inlayHints.typeHints` - enable hints for inferred types.
* `rust-analyzer.inlayHints.chainingHints` - enable hints for inferred types on method chains.
* `rust-analyzer.inlayHints.parameterHints` - enable hints for function parameters.
* `rust-analyzer.inlayHints.maxLength` — shortens the hints if their length exceeds the value specified. If no value is specified (`null`), no shortening is applied.

**Note:** VS Code does not have native support for inlay hints [yet](https://github.com/microsoft/vscode/issues/16221) and the hints are implemented using decorations.
This approach has limitations, the caret movement and bracket highlighting near the edges of the hint may be weird:
[1](https://github.com/rust-analyzer/rust-analyzer/issues/1623), [2](https://github.com/rust-analyzer/rust-analyzer/issues/3453).
