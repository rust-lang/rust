This document is an index of features that the rust-analyzer language server
provides. Shortcuts are for the default VS Code layout. If there's no shortcut,
you can use <kbd>Ctrl+Shift+P</kbd> to search for the corresponding action.

### Commands <kbd>ctrl+shift+p</kbd>


#### Toggle inlay hints

Toggle inlay hints view for the current workspace.
It is recommended to assign a shortcut for this command to quickly turn off
inlay hints when they prevent you from reading/writing the code.

### Magic Completions

In addition to usual reference completion, rust-analyzer provides some ✨magic✨
completions as well:

Keywords like `if`, `else` `while`, `loop` are completed with braces, and cursor
is placed at the appropriate position. Even though `if` is easy to type, you
still want to complete it, to get ` { }` for free! `return` is inserted with a
space or `;` depending on the return type of the function.

When completing a function call, `()` are automatically inserted. If a function
takes arguments, the cursor is positioned inside the parenthesis.

There are postfix completions, which can be triggered by typing something like
`foo().if`. The word after `.` determines postfix completion. Possible variants are:

- `expr.if` -> `if expr {}` or `if let ... {}` for `Option` or `Result`
- `expr.match` -> `match expr {}`
- `expr.while` -> `while expr {}` or `while let ... {}` for `Option` or `Result`
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
- `tmod` ->
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fn() {}
}
```

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
