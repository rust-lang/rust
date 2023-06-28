## Statements

### Let statements

Put a space after the `:` and on both sides of the `=` (if they are present).
Don't put a space before the semicolon.

```rust
// A comment.
let pattern: Type = expr;

let pattern;
let pattern: Type;
let pattern = expr;
```

If possible, format the declaration on a single line. If not possible, then try
splitting after the `=`, if the declaration fits on two lines. Block-indent the
expression.

```rust
let pattern: Type =
    expr;
```

If the first line still does not fit on a single line, split after the `:`, and
use block indentation. If the type requires multiple lines, even after
line-breaking after the `:`, then place the first line on the same line as the
`:`, subject to the [combining rules](expressions.html#combinable-expressions).


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
fits in the remaining space, it stays on the same line as the `=`, and the rest
of the expression is not further indented. If the first line does not fit, then
put the expression on subsequent lines, block indented. If the expression is a
block and the type or pattern cover multiple lines, put the opening brace on a
new line and not indented (this provides separation for the interior of the
block from the type); otherwise, the opening brace follows the `=`.

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

A let statement can contain an `else` component, making it a let-else statement.
In this case, always apply the same formatting rules to the components preceding
the `else` block (i.e. the `let pattern: Type = initializer_expr` portion)
as described [for other let statements](#let-statements).

Format the entire let-else statement on a single line if all the following are
true:

* the entire statement is *short*
* the `else` block contains only a single-line expression and no statements
* the `else` block contains no comments
* the let statement components preceding the `else` block can be formatted on a single line

```rust
let Some(1) = opt else { return };
```

Otherwise, the let-else statement requires some line breaks.

If breaking a let-else statement across multiple lines, never break between the
`else` and the `{`, and always break before the `}`.

If the let statement components preceding the `else` can be formatted on a
single line, but the let-else does not qualify to be placed entirely on a
single line, put the `else {` on the same line as the initializer expression,
with a space between them, then break the line after the `{`. Indent the
closing `}` to match the `let`, and indent the contained block one step
further.

```rust
let Some(1) = opt else {
    return;
};

let Some(1) = opt else {
    // nope
    return
};
```

If the let statement components preceding the `else` can be formatted on a
single line, but the `else {` does not fit on the same line, break the line
before the `else`.

```rust
    let Some(x) = some_really_really_really_really_really_really_really_really_really_long_name
    else {
        return;
    };
```

If the initializer expression is multi-line, put the `else` keyword and opening
brace of the block (i.e. `else {`) on the same line as the end of the
initializer expression, with a space between them, if and only if all the
following are true:

* The initializer expression ends with one or more closing
  parentheses, square brackets, and/or braces
* There is nothing else on that line
* That line has the same indentation level as the initial `let` keyword.

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

Otherwise, put the `else` keyword and opening brace on the next line after the
end of the initializer expression, with the `else` keyword at the same
indentation level as the `let` keyword.

For example:

```rust
fn main() {
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
        return
    };

    let Some(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) =
        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    else {
        return;
    };

    let LongStructName(AnotherStruct {
        multi,
        line,
        pattern,
    }) = slice.as_ref()
    else {
        return;
    };

    let LongStructName(AnotherStruct {
        multi,
        line,
        pattern,
    }) = multi_line_function_call(
        arg1,
        arg2,
        arg3,
        arg4,
    ) else {
        return;
    };
}
```

### Macros in statement position

For a macro use in statement position, use parentheses or square brackets as
delimiters, and terminate it with a semicolon. Do not put spaces around the
name, `!`, the delimiters, or the `;`.

```rust
// A comment.
a_macro!(...);
```


### Expressions in statement position

Do not put space between the expression and the semicolon.

```
<expr>;
```

Terminate all expressions in statement position with a semicolon, unless they
end with a block or are used as the value for a block.

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

Use a semicolon where an expression has void type, even if it could be
propagated. E.g.,

```rust
fn foo() { ... }

fn bar() {
    foo();
}
```
