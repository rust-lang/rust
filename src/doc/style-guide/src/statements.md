### Let statements

There should be spaces after the `:` and on both sides of the `=` (if they are
present). No space before the semi-colon.

```rust
// A comment.
let pattern: Type = expr;

let pattern;
let pattern: Type;
let pattern = expr;
```

If possible the declaration should be formatted on a single line. If this is not
possible, then try splitting after the `=`, if the declaration can fit on two
lines. The expression should be block indented.

```rust
let pattern: Type =
    expr;
```

If the first line does not fit on a single line, then split after the colon,
using block indentation. If the type covers multiple lines, even after line-
breaking after the `:`, then the first line may be placed on the same line as
the `:`, subject to the [combining rules](https://github.com/rust-lang-nursery/fmt-rfcs/issues/61) (WIP).


```rust
let pattern:
    Type =
    expr;
```

e.g,

```rust
let Foo {
    f: abcd,
    g: qwer,
}: Foo<Bar> =
    Foo { f, g };

let (abcd,
    defg):
    Baz =
{ ... }
```

If the expression covers multiple lines, if the first line of the expression
fits in the remaining space, it stays on the same line as the `=`, the rest of the
expression is not indented. If the first line does not fit, then it should start
on the next lines, and should be block indented. If the expression is a block
and the type or pattern cover multiple lines, then the opening brace should be
on a new line and not indented (this provides separation for the interior of the
block from the type), otherwise the opening brace follows the `=`.

Examples:

```rust
let foo = Foo {
    f: abcd,
    g: qwer,
};

let foo =
    ALongName {
        f: abcd,
        g: qwer,
    };

let foo: Type = {
    an_expression();
    ...
};

let foo:
    ALongType =
{
    an_expression();
    ...
};

let Foo {
    f: abcd,
    g: qwer,
}: Foo<Bar> = Foo {
    f: blimblimblim,
    g: blamblamblam,
};

let Foo {
    f: abcd,
    g: qwer,
}: Foo<Bar> = foo(
    blimblimblim,
    blamblamblam,
);
```


### Macros in statement position

A macro use in statement position should use parentheses or square brackets as
delimiters and should be terminated with a semi-colon. There should be no spaces
between the name, `!`, the delimiters, or the `;`.

```rust
// A comment.
a_macro!(...);
```


### Expressions in statement position

There should be no space between the expression and the semi-colon.

```
<expr>;
```

All expressions in statement position should be terminated with a semi-colon,
unless they end with a block or are used as the value for a block.

E.g.,

```rust
{
    an_expression();
    expr_as_value()
}

return foo();

loop {
    break;
}
```

Use a semi-colon where an expression has void type, even if it could be
propagated. E.g.,

```rust
fn foo() { ... }

fn bar() {
    foo();
}
```
