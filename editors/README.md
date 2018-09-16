To install experimental VS Code plugin:

```
$ cargo install-code
```

This will run `cargo install --packge ra_lsp_server` to install the
server binary into `~/.cargo/bin`, and then will build and install
plugin from `editors/code`. See
[this](https://github.com/matklad/rust-analyzer/blob/cc76b0d31d8ba013c499dd3a4ca69b37004795e6/crates/tools/src/main.rs#L192)
for details

It's better to remove existing Rust plugins to avoid interference.

### Features:

* syntax highlighting (LSP does not have API for it, so impl is hacky
  and sometimes fall-backs to the horrible built-in highlighting)
  
* **Go to symbol in workspace** (`ctrl+t`)
  - `#Foo` searches for `Foo` type in the current workspace
  - `#foo#` searches for `foo` function in the current workspace
  - `#Foo*` searches for `Foo` type among dependencies, excluding `stdlib`
  - Sorry for a weired UI, neither LSP, not VSCode have any sane API for filtering! :)

* **Go to symbol in file** (`alt+shift+o`)

* **Go to definition** ("correct" for `mod foo;` decls, approximate for other things).

* commands (`ctrl+shift+p` or keybindings)
  - **Show Rust Syntax Tree** (use it to verify that plugin works)
  - **Rust Extend Selection**. Extends the current selection to the
    encompassing syntactic construct (expression, statement, item,
    module, etc). It works with multiple cursors. Do bind this command
    to a key, its super-useful!
  - **Rust Matching Brace**. If the cursor is on any brace
    (`<>(){}[]`) which is a part of a brace-pair, moves cursor to the
    matching brace.
  - **Rust Parent Module**. Navigate to the parent module of the current module
  - **Rust Join Lines**. Join selected lines into one, smartly fixing
    up whitespace and trailing commas.
  - **Run test at caret**. When cursor is inside a function marked
    `#[test]`, this action runs this specific test. If the cursor is
    outside of the test function, this re-runs the last test. Do bind
    this to a shortcut!

* code actions (use `ctrl+.` to activate).

`<|>` signifies cursor position

- Flip `,`

```
// before:
fn foo(x: usize,<|> dim: (usize, usize))
// after:
fn foo(dim: (usize, usize), x: usize)
```

- Add `#[derive]`

```
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

```
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
