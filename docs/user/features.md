This documents is an index of features that rust-analyzer language server provides.

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


