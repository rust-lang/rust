## Expressions

### Blocks

A block expression must have a newline after the initial `{` and before the
terminal `}`, unless it qualifies to be written as a single line based on
another style rule.

A keyword before the block (such as `unsafe` or `async`) must be on the same
line as the opening brace, with a single space between the keyword and the
opening brace. Indent the contents of the block.

```rust
fn block_as_stmt() {
    a_call();

    {
        a_call_inside_a_block();

        // a comment in a block
        the_value
    }
}

fn block_as_expr() {
    let foo = {
        a_call_inside_a_block();

        // a comment in a block
        the_value
    };
}

fn unsafe_block_as_stmt() {
    a_call();

    unsafe {
        a_call_inside_a_block();

        // a comment in a block
        the_value
    }
}
```

If a block has an attribute, put it on its own line before the block:

```rust
fn block_as_stmt() {
    #[an_attribute]
    {
        #![an_inner_attribute]

        // a comment in a block
        the_value
    }
}
```

Avoid writing comments on the same lines as either of the braces.

Write an empty block as `{}`.

Write a block on a single line if:

* it is either used in expression position (not statement position) or is an
  unsafe block in statement position,
* it contains a single-line expression and no statements, and
* it contains no comments

For a single-line block, put spaces after the opening brace and before the
closing brace.

Examples:

```rust
fn main() {
    // Single line
    let _ = { a_call() };
    let _ = unsafe { a_call() };

    // Not allowed on one line
    // Statement position.
    {
        a_call()
    }

    // Contains a statement
    let _ = {
        a_call();
    };
    unsafe {
        a_call();
    }

    // Contains a comment
    let _ = {
        // A comment
    };
    let _ = {
        // A comment
        a_call()
    };

    // Multiple lines
    let _ = {
        a_call();
        another_call()
    };
    let _ = {
        a_call(
            an_argument,
            another_arg,
        )
    };
}
```


### Closures

Don't put any extra spaces before the first `|` (unless the closure is prefixed
by a keyword such as `move`); put a space between the second `|` and the
expression of the closure. Between the `|`s, use function definition syntax,
but elide types where possible.

Use closures without the enclosing `{}`, if possible. Add the `{}` when you have
a return type, when there are statements, when there are comments inside the
closure, or when the body expression is a control-flow expression that spans
multiple lines. If using braces, follow the rules above for blocks. Examples:

```rust
|arg1, arg2| expr

move |arg1: i32, arg2: i32| -> i32 {
    expr1;
    expr2
}

|| Foo {
    field1,
    field2: 0,
}

|| {
    if true {
        blah
    } else {
        boo
    }
}

|x| unsafe {
    expr
}
```


### Struct literals

If a struct literal is *small*, format it on a single line, and do not use a
trailing comma. If not, split it across multiple lines, with each field on its
own block-indented line, and use a trailing comma.

For each `field: value` entry, put a space after the colon only.

Put a space before the opening brace. In the single-line form, put spaces after
the opening brace and before the closing brace.

```rust
Foo { field1, field2: 0 }
let f = Foo {
    field1,
    field2: an_expr,
};
```

Functional record update syntax is treated like a field, but it must never have
a trailing comma. Do not put a space after `..`.

```rust
let f = Foo {
    field1,
    ..an_expr
};
```


### Tuple literals

Use a single-line form where possible. Do not put spaces between the opening
parenthesis and the first element, or between the last element and the closing
parenthesis. Separate elements with a comma followed by a space.

Where a single-line form is not possible, write the tuple across
multiple lines, with each element of the tuple on its own block-indented line,
and use a trailing comma.

```rust
(a, b, c)

let x = (
    a_long_expr,
    another_very_long_expr,
);
```


### Tuple struct literals

Do not put space between the identifier and the opening parenthesis. Otherwise,
follow the rules for tuple literals:

```rust
Foo(a, b, c)

let x = Foo(
    a_long_expr,
    another_very_long_expr,
);
```


### Enum literals

Follow the formatting rules for the various struct literals. Prefer using the
name of the enum as a qualifying name, unless the enum is in the prelude:

```rust
Foo::Bar(a, b)
Foo::Baz {
    field1,
    field2: 1001,
}
Ok(an_expr)
```


### Array literals

Write small array literals on a single line. Do not put spaces between the opening
square bracket and the first element, or between the last element and the closing
square bracket. Separate elements with a comma followed by a space.

If using the repeating initializer, put a space after the semicolon
only.

Apply the same rules if using `vec!` or similar array-like macros; always use
square brackets with such macros. Examples:

```rust
fn main() {
    let x = [1, 2, 3];
    let y = vec![a, b, c, d];
    let a = [42; 10];
}
```

For arrays that have to be broken across lines, if using the repeating
initializer, break after the `;`, not before. Otherwise, follow the rules below
for function calls. In any case, block-indent the contents of the initializer,
and put line breaks after the opening square bracket and before the closing
square bracket:

```rust
fn main() {
    [
        a_long_expression();
        1234567890
    ]
    let x = [
        an_expression,
        another_expression,
        a_third_expression,
    ];
}
```


### Array accesses, indexing, and slicing.

Don't put spaces around the square brackets. Avoid breaking lines if possible.
Never break a line between the target expression and the opening square
bracket. If the indexing expression must be broken onto a subsequent line, or
spans multiple lines itself, then block-indent the indexing expression, and put
newlines after the opening square bracket and before the closing square
bracket:

Examples:

```rust
fn main() {
    foo[42];
    &foo[..10];
    bar[0..100];
    foo[4 + 5 / bar];
    a_long_target[
        a_long_indexing_expression
    ];
}
```

### Unary operations

Do not include a space between a unary op and its operand (i.e., `!x`, not
`! x`). However, there must be a space after `&mut`. Avoid line-breaking
between a unary operator and its operand.

### Binary operations

Do include spaces around binary ops (i.e., `x + 1`, not `x+1`) (including `=`
and other assignment operators such as `+=` or `*=`).

For comparison operators, because for `T op U`, `&T op &U` is also implemented:
if you have `t: &T`, and `u: U`, prefer `*t op u` to `t op &u`. In general,
within expressions, prefer dereferencing to taking references, unless necessary
(e.g. to avoid an unnecessarily expensive operation).

Use parentheses liberally; do not necessarily elide them due to precedence.
Tools should not automatically insert or remove parentheses. Do not use spaces
to indicate precedence.

If line-breaking, block-indent each subsequent line. For assignment operators,
break after the operator; for all other operators, put the operator on the
subsequent line. Put each sub-expression on its own line:

```rust
foo_bar
    + bar
    + baz
    + qux
    + whatever
```

Prefer line-breaking at an assignment operator (either `=` or `+=`, etc.) rather
than at other binary operators.

### Control flow

Do not include extraneous parentheses for `if` and `while` expressions.

```rust
if true {
}
```

is better than

```rust
if (true) {
}
```

Do include extraneous parentheses if it makes an arithmetic or logic expression
easier to understand (`(x * 15) + (y * 20)` is fine)

### Function calls

Do not put a space between the function name, and the opening parenthesis.

Do not put a space between an argument, and the comma which follows.

Do put a space between an argument, and the comma which precedes it.

Prefer not to break a line in the callee expression.

#### Single-line calls

Do not put a space between the function name and open paren, between the open
paren and the first argument, or between the last argument and the close paren.

Do not put a comma after the last argument.

```rust
foo(x, y, z)
```

#### Multi-line calls

If the function call is not *small*, it would otherwise over-run the max width,
or any argument or the callee is multi-line, then format the call across
multiple lines. In this case, put each argument on its own block-indented line,
break after the opening parenthesis and before the closing parenthesis,
and use a trailing comma:

```rust
a_function_call(
    arg1,
    a_nested_call(a, b),
)
```


### Method calls

Follow the function rules for calling.

Do not put any spaces around the `.`.

```rust
x.foo().bar().baz(x, y, z);
```


### Macro uses

If a macro can be parsed like other constructs, format it like those
constructs. For example, a macro use `foo!(a, b, c)` can be parsed like a
function call (ignoring the `!`), so format it using the rules for function
calls.

#### Special case macros

For macros which take a format string, if all other arguments are *small*,
format the arguments before the format string on a single line if they fit, and
format the arguments after the format string on a single line if they fit, with
the format string on its own line. If the arguments are not small or do not
fit, put each on its own line as with a function. For example:

```rust
println!(
    "Hello {} and {}",
    name1, name2,
);

assert_eq!(
    x, y,
    "x and y were not equal, see {}",
    reason,
);
```


### Casts (`as`)

Put spaces before and after `as`:

```rust
let cstr = "Hi\0" as *const str as *const [u8] as *const std::os::raw::c_char;
```


### Chains of fields and method calls

A chain is a sequence of field accesses, method calls, and/or uses of the try
operator `?`. E.g., `a.b.c().d` or `foo?.bar().baz?`.

Format the chain on one line if it is "small" and otherwise possible to do so.
If formatting on multiple lines, put each field access or method call in the
chain on its own line, with the line-break before the `.` and after any `?`.
Block-indent each subsequent line:

```rust
let foo = bar
    .baz?
    .qux();
```

If the length of the last line of the first element plus its indentation is
less than or equal to the indentation of the second line, then combine the
first and second lines if they fit. Apply this rule recursively.

```rust
x.baz?
    .qux()

x.y.z
    .qux()

let foo = x
    .baz?
    .qux();

foo(
    expr1,
    expr2,
).baz?
    .qux();
```

#### Multi-line elements

If any element in a chain is formatted across multiple lines, put that element
and any later elements on their own lines.

```rust
a.b.c()?
    .foo(
        an_expr,
        another_expr,
    )
    .bar
    .baz
```

Note there is block indent due to the chain and the function call in the above
example.

Prefer formatting the whole chain in multi-line style and each element on one
line, rather than putting some elements on multiple lines and some on a single
line, e.g.,

```rust
// Better
self.pre_comment
    .as_ref()
    .map_or(false, |comment| comment.starts_with("//"))

// Worse
self.pre_comment.as_ref().map_or(
    false,
    |comment| comment.starts_with("//"),
)
```

### Control flow expressions

This section covers `if`, `if let`, `loop`, `while`, `while let`, and `for`
expressions.

Put the keyword, any initial clauses, and the opening brace of the block all on
a single line, if they fit. Apply the usual rules for [block
formatting](#blocks) to the block.

If there is an `else` component, then put the closing brace, `else`, any
following clause, and the opening brace all on the same line, with a single
space before and after the `else` keyword:

```rust
if ... {
    ...
} else {
    ...
}

if let ... {
    ...
} else if ... {
    ...
} else {
    ...
}
```

If the control line needs to be broken, prefer to break before the `=` in `*
let` expressions and before `in` in a `for` expression; block-indent the
following line. If the control line is broken for any reason, put the opening
brace on its own line, not indented. Examples:

```rust
while let Some(foo)
    = a_long_expression
{
    ...
}

for foo
    in a_long_expression
{
    ...
}

if a_long_expression
    && another_long_expression
    || a_third_long_expression
{
    ...
}
```

Where the initial clause spans multiple lines and ends with one or more closing
parentheses, square brackets, or braces, and there is nothing else on that
line, and that line is not indented beyond the indent on the first line of the
control flow expression, then put the opening brace of the block on the same
line with a preceding space. For example:

```rust
if !self.config.file_lines().intersects(
    &self.codemap.lookup_line_range(
        stmt.span,
    ),
) {  // Opening brace on same line as initial clause.
    ...
}
```


#### Single line `if else`

Put an `if else` or `if let else` on a single line if it occurs in expression
context (i.e., is not a standalone statement), it contains a single `else`
clause, and is *small*:

```rust
let y = if x { 0 } else { 1 };

// Examples that must be multi-line.
let y = if something_very_long {
    not_small
} else {
    also_not_small
};

if x {
    0
} else {
    1
}
```


### Match

Prefer not to line-break inside the discriminant expression. Always break after
the opening brace and before the closing brace. Block-indent the match arms
once:

```rust
match foo {
    // arms
}

let x = match foo.bar.baz() {
    // arms
};
```

Use a trailing comma for a match arm if and only if not using a block.

Never start a match arm pattern with `|`:

```rust
match foo {
    // Don't do this.
    | foo => bar,
    // Or this.
    | a_very_long_pattern
    | another_pattern
    | yet_another_pattern
    | a_fourth_pattern => {
        ...
    }
}
```

Prefer:

```rust
match foo {
    foo => bar,
    a_very_long_pattern
    | another_pattern
    | yet_another_pattern
    | a_fourth_pattern => {
        ...
    }
}
```

Avoid splitting the left-hand side (before the `=>`) of a match arm where
possible. If the right-hand side of the match arm is kept on the same line,
never use a block (unless the block is empty).

If the right-hand side consists of multiple statements, or has line comments,
or the start of the line does not fit on the same line as the left-hand side,
use a block.

Block-indent the body of a block arm.

Examples:

```rust
match foo {
    foo => bar,
    a_very_long_pattern | another_pattern if an_expression() => {
        no_room_for_this_expression()
    }
    foo => {
        // A comment.
        an_expression()
    }
    foo => {
        let a = statement();
        an_expression()
    }
    bar => {}
    // Trailing comma on last item.
    foo => bar,
}
```

If the body is a single expression with no line comments and not a control flow
expression, start it on the same line as the left-hand side. If not, then it
must be in a block. Example:

```rust
match foo {
    // A combinable expression.
    foo => a_function_call(another_call(
        argument1,
        argument2,
    )),
    // A non-combinable expression
    bar => {
        a_function_call(
            another_call(
                argument1,
                argument2,
            ),
            another_argument,
        )
    }
}
```

#### Line-breaking

If using a block form on the right-hand side of a match arm makes it possible
to avoid breaking on the left-hand side, do that:

```rust
    // Assuming the following line does not fit in the max width
    a_very_long_pattern | another_pattern => ALongStructName {
        ...
    },
    // Prefer this
    a_very_long_pattern | another_pattern => {
        ALongStructName {
            ...
        }
    }
    // To splitting the pattern.
```

Never break after `=>` without using the block form of the body.

If the left-hand side must be split and there is an `if` clause, break before
the `if` and block indent. In this case, always use a block body and start the
body on a new line:

```rust
    a_very_long_pattern | another_pattern
        if expr =>
    {
        ...
    }
```

If required to break the pattern, put each clause of the pattern on its own
line with no additional indent, breaking before the `|`. If there is an `if`
clause, use the above form:

```rust
    a_very_long_pattern
    | another_pattern
    | yet_another_pattern
    | a_forth_pattern => {
        ...
    }
    a_very_long_pattern
    | another_pattern
    | yet_another_pattern
    | a_forth_pattern
        if expr =>
    {
        ...
    }
```

If the pattern is multi-line, and the last line is less wide than the indent, do
not put the `if` clause on a new line. E.g.,

```rust
    Token::Dimension {
         value,
         ref unit,
         ..
    } if num_context.is_ok(context.parsing_mode, value) => {
        ...
    }
```

If every clause in a pattern is *small*, but the whole pattern does not fit on
one line, then format the pattern across multiple lines with as many clauses
per line as possible. Again, break before a `|`:

```rust
    foo | bar | baz
    | qux => {
        ...
    }
```

We define a pattern clause to be *small* if it fits on a single line and
matches "small" in the following grammar:

```
small:
    - small_no_tuple
    - unary tuple constructor: `(` small_no_tuple `,` `)`
    - `&` small

small_no_tuple:
    - single token
    - `&` small_no_tuple
```

E.g., `&&Some(foo)` matches, `Foo(4, Bar)` does not.


### Combinable expressions

Where a function call has a single argument, and that argument is formatted
across multiple-lines, format the outer call as if it were a single-line call,
if the result fits. Apply the same combining behaviour to any similar
expressions which have multi-line, block-indented lists of sub-expressions
delimited by parentheses (e.g., macros or tuple struct literals). E.g.,

```rust
foo(bar(
    an_expr,
    another_expr,
))

let x = foo(Bar {
    field: whatever,
});

foo(|param| {
    action();
    foo(param)
})

let x = combinable([
    an_expr,
    another_expr,
]);

let arr = [combinable(
    an_expr,
    another_expr,
)];
```

Apply this behavior recursively.

For a function with multiple arguments, if the last argument is a multi-line
closure with an explicit block, there are no other closure arguments, and all
the arguments and the first line of the closure fit on the first line, use the
same combining behavior:

```rust
foo(first_arg, x, |param| {
    action();
    foo(param)
})
```


### Ranges

Do not put spaces in ranges, e.g., `0..10`, `x..=y`, `..x.len()`, `foo..`.

When writing a range with both upper and lower bounds, if the line must be
broken within the range, break before the range operator and block indent the
second line:

```rust
a_long_expression
    ..another_long_expression
```

For the sake of indicating precedence, if either bound is a compound
expression, use parentheses around it, e.g., `..(x + 1)`, `(x.f)..(x.f.len())`,
or `0..(x - 10)`.


### Hexadecimal literals

Hexadecimal literals may use upper- or lower-case letters, but they must not be
mixed within the same literal. Projects should use the same case for all
literals, but we do not make a recommendation for either lower- or upper-case.

## Patterns

Format patterns like their corresponding expressions. See the section on
`match` for additional formatting for patterns in match arms.
