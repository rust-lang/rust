
Preqrequisites:

In order to build the VS Code plugin, you need to have node.js and npm with
a minimum version of 10 installed. Please refer to
[node.js and npm documentation](https://nodejs.org) for installation instructions.

The experimental VS Code plugin can then be built and installed by executing the
following commands:

```
$ git clone https://github.com/rust-analyzer/rust-analyzer.git --depth 1
$ cd rust-analyzer
$ cargo install-code

# for stdlib support
$ rustup component add rust-src
```

This will run `cargo install --package ra_lsp_server` to install the server
binary into `~/.cargo/bin`, and then will build and install plugin from
`editors/code`. See
[this](https://github.com/rust-analyzer/rust-analyzer/blob/0199572a3d06ff66eeae85a2d2c9762996f0d2d8/crates/tools/src/main.rs#L150)
for details. The installation is expected to *just work*, if it doesn't, report
bugs!

It's better to remove existing Rust plugins to avoid interference.

## Rust Analyzer Specific Features

These features are implemented as extensions to the language server protocol.
They are more experimental in nature and work only with VS Code.

### Syntax highlighting

It overrides built-in highlighting, and works only with a specific theme
(zenburn). `rust-analyzer.highlightingOn` setting can be used to disable it.

### Go to symbol in workspace <kbd>ctrl+t</kbd>

It mostly works on top of the built-in LSP functionality, however `#` and `*`
symbols can be used to narrow down the search. Specifically,

- `#Foo` searches for `Foo` type in the current workspace
- `#foo#` searches for `foo` function in the current workspace
- `#Foo*` searches for `Foo` type among dependencies, excluding `stdlib`
- `#foo#*` searches for `foo` function among dependencies.

That is, `#` switches from "types" to all symbols, `*` switches from the current
workspace to dependencies.

### Commands <kbd>ctrl+shift+p</kbd>

#### Show Rust Syntax Tree

Shows the parse tree of the current file. It exists mostly for debugging
rust-analyzer itself.

#### Extend Selection

Extends the current selection to the encompassing syntactic construct
(expression, statement, item, module, etc). It works with multiple cursors. Do
bind this command to a key, its super-useful! Expected to be upstreamed to LSP soonish:
https://github.com/Microsoft/language-server-protocol/issues/613

#### Matching Brace

If the cursor is on any brace (`<>(){}[]`) which is a part of a brace-pair,
moves cursor to the matching brace. It uses the actual parser to determine
braces, so it won't confuse generics with comparisons.

#### Parent Module

Navigates to the parent module of the current module.

#### Join Lines

Join selected lines into one, smartly fixing up whitespace and trailing commas.

#### Run

Shows popup suggesting to run a test/benchmark/binary **at the current cursor
location**. Super useful for repeatedly running just a single test. Do bind this
to a shortcut!


### On Typing Assists

Some features trigger on typing certain characters:

- typing `let =` tries to smartly add `;` if `=` is followed by an existing expression.
- Enter inside comments automatically inserts `///`
- typing `.` in a chain method call auto-indents


### Code Actions (Assists)

These are triggered in a particular context via light bulb. We use custom code on
the VS Code side to be able to position cursor.


- Flip `,`

```rust
// before:
fn foo(x: usize,<|> dim: (usize, usize))
// after:
fn foo(dim: (usize, usize), x: usize)
```

- Add `#[derive]`

```rust
// before:
struct Foo {
    <|>x: i32
}
// after:
#[derive(<|>)]
struct Foo {
    x: i32
}
```

- Add `impl`

```rust
// before:
struct Foo<'a, T: Debug> {
    <|>t: T
}
// after:
struct Foo<'a, T: Debug> {
    t: T
}

impl<'a, T: Debug> Foo<'a, T> {
    <|>
}
```

- Change visibility

```rust
// before:
fn<|> foo() {}

// after
pub(crate) fn foo() {}
```

- Introduce variable:

```rust
// before:
fn foo() {
    foo(<|>1 + 1<|>);
}

// after:
fn foo() {
    let var_name = 1 + 1;
    foo(var_name);
}
```

- Replace if-let with match:

```rust
// before:
impl VariantData {
    pub fn is_struct(&self) -> bool {
        if <|>let VariantData::Struct(..) = *self {
            true
        } else {
            false
        }
    }
}

// after:
impl VariantData {
    pub fn is_struct(&self) -> bool {
        <|>match *self {
            VariantData::Struct(..) => true,
            _ => false,
        }
    }
}
```

- Split import

```rust
// before:
use algo:<|>:visitor::{Visitor, visit};
//after:
use algo::{<|>visitor::{Visitor, visit}};
```

## LSP features

* **Go to definition**: works correctly for local variables and some paths,
  falls back to heuristic name matching for other things for the time being.

* **Completion**: completes paths, including dependencies and standard library.
  Does not handle glob imports and macros. Completes fields and inherent
  methods.

* **Outline** <kbd>alt+shift+o</kbd>

* **Signature Info**

* **Format document**. Formats the current file with rustfmt. Rustfmt must be
  installed separately with `rustup component add rustfmt`.

* **Hover** shows types of expressions and docstings

* **Rename** works for local variables

* **Code Lens** for running tests

* **Folding**

* **Diagnostics**
  - missing module for `mod foo;` with a fix to create `foo.rs`.
  - struct field shorthand
  - unnecessary braces in use item


## Performance

Rust Analyzer is expected to be pretty fast. Specifically, the initial analysis
of the project (i.e, when you first invoke completion or symbols) typically
takes dozen of seconds at most. After that, everything is supposed to be more or
less instant. However currently all analysis results are kept in memory, so
memory usage is pretty high. Working with `rust-lang/rust` repo, for example,
needs about 5 gigabytes of ram.
