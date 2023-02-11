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

#### else blocks (let-else statements)

If a let statement contains an `else` component, also known as a let-else statement,
then the `else` component should be formatted according to the same rules as the `else` block
in [control flow expressions (i.e. if-else, and if-let-else expressions)](./expressions.md#control-flow-expressions).
Apply the same formatting rules to the components preceding
the `else` block (i.e. the `let pattern: Type = initializer_expr ...` portion)
as described [above](#let-statements)

Similarly to if-else expressions, if the initializer
expression is multi-lined, then the `else` keyword and opening brace of the block (i.e. `else {`)
should be put on the same line as the end of the initializer
expression with a preceding space if all the following are true:

* The initializer expression ends with one or more closing
  parentheses, square brackets, and/or braces
* There is nothing else on that line
* That line is not indented beyond the indent of the first line containing the `let` keyword

For example:

```rust
let Some(x) = y.foo(
    "abc",
    fairly_long_identifier,
    "def",
    "123456",
    "string",
    "cheese",
) else {
    bar()
}
```

Otherwise, the `else` keyword and opening brace should be placed on the next line after the end of the initializer expression, and should not be indented (the `else` keyword should be aligned with the `let` keyword).

For example:

```rust
let Some(x) = abcdef()
    .foo(
        "abc",
        some_really_really_really_long_ident,
        "ident",
        "123456",
    )
    .bar()
    .baz()
    .qux("fffffffffffffffff")
else {
    foo_bar()
}
```

##### Single line let-else statements

The entire let-else statement may be formatted on a single line if all the following are true:

* the entire statement is *short*
* the `else` block contains a single-line expression and no statements
* the `else` block contains no comments
* the let statement components preceding the `else` block can be formatted on a single line

```rust
let Some(1) = opt else { return };

let Some(1) = opt else {
    return;
};

let Some(1) = opt else {
    // nope
    return
};
```

Formatters may allow users to configure the value of the threshold
used to determine whether a let-else statement is *short*.

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
