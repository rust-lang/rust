## Expressions

### Blocks

A block expression should have a newline after the initial `{` and before the
terminal `}`. Any qualifier before the block (e.g., `unsafe`) should always be
on the same line as the opening brace, and separated with a single space. The
contents of the block should be block indented:

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

If a block has an attribute, it should be on its own line:

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

Avoid writing comments on the same line as the braces.

An empty block should be written as `{}`.

A block may be written on a single line if:

* it is either used in expression position (not statement position) or is an
  unsafe block in statement position
* contains a single-line expression and no statements
* contains no comments

A single line block should have spaces after the opening brace and before the
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
by `move`); put a space between the second `|` and the expression of the
closure. Between the `|`s, you should use function definition syntax, however,
elide types where possible.

Use closures without the enclosing `{}`, if possible. Add the `{}` when you have
a return type, when there are statements, there are comments in the body, or the
body expression spans multiple lines and is a control-flow expression. If using
braces, follow the rules above for blocks. Examples:

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

If a struct literal is *small* it may be formatted on a single line. If not,
each field should be on it's own, block-indented line. There should be a
trailing comma in the multi-line form only. There should be a space after the
colon only.

There should be a space before the opening brace. In the single-line form there
should be spaces after the opening brace and before the closing brace.

```rust
Foo { field1, field2: 0 }
let f = Foo {
    field1,
    field2: an_expr,
};
```

Functional record update syntax is treated like a field, but it must never have
a trailing comma. There should be no space after `..`.

let f = Foo {
    field1,
    ..an_expr
};


### Tuple literals

Use a single-line form where possible. There should not be spaces around the
parentheses. Where a single-line form is not possible, each element of the tuple
should be on its own block-indented line and there should be a trailing comma.

```rust
(a, b, c)

let x = (
    a_long_expr,
    another_very_long_expr,
);
```


### Tuple struct literals

There should be no space between the identifier and the opening parenthesis.
Otherwise, follow the rules for tuple literals, e.g., `Foo(a, b)`.


### Enum literals

Follow the formatting rules for the various struct literals. Prefer using the
name of the enum as a qualifying name, unless the enum is in the prelude. E.g.,

```rust
Foo::Bar(a, b)
Foo::Baz {
    field1,
    field2: 1001,
}
Ok(an_expr)
```


### Array literals

For simple array literals, avoid line breaking, no spaces around square
brackets, contents of the array should be separated by commas and spaces. If
using the repeating initialiser, there should be a space after the semicolon
only. Apply the same rules if using the `vec!` or similar macros (always use
square brackets here). Examples:

```rust
fn main() {
    [1, 2, 3];
    vec![a, b, c, d];
    let a = [42; 10];
}
```

If a line must be broken, prefer breaking only after the `;`, if possible.
Otherwise, follow the rules below for function calls. In any case, the contents
of the initialiser should be block indented and there should be line breaks
after the opening bracket and before the closing bracket:

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

No spaces around the square brackets, avoid breaking lines if possible, never
break a line between the target expression and the opening bracket. If the
indexing expression covers multiple lines, then it should be block indented and
there should be newlines after the opening brackets and before the closing
bracket. However, this should be avoided where possible.

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

Use parentheses liberally, do not necessarily elide them due to precedence.
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
or any argument or the callee is multi-line, then the call should be formatted
across multiple lines. In this case, each argument should be on it's own block-
indented line, there should be a newline after the opening parenthesis and
before the closing parenthesis, and there should be a trailing comma. E.g.,

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

Macros which can be parsed like other constructs should be formatted like those
constructs. For example, a macro use `foo!(a, b, c)` can be parsed like a
function call (ignoring the `!`), therefore it should be formatted following the
rules for function calls.

#### Special case macros

Macros which take a format string and where all other arguments are *small* may
be formatted with arguments before and after the format string on a single line
and the format string on its own line, rather than putting each argument on its
own line. For example,

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

A chain is a sequence of field accesses and/or method calls. A chain may also
include the try operator ('?'). E.g., `a.b.c().d` or `foo?.bar().baz?`.

Prefer formatting on one line if possible, and the chain is *small*. If
formatting on multiple lines, each field access or method call in the chain
should be on its own line with the line-break before the `.` and after any `?`.
Each line should be block-indented. E.g.,

```rust
let foo = bar
    .baz?
    .qux();
```

If the length of the last line of the first element plus its indentation is
less than or equal to the indentation of the second line (and there is space),
then combine the first and second lines, e.g.,

```rust
x.baz?
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

The keyword, any initial clauses, and the opening brace of the block should be
on a single line. The usual rules for [block formatting](#blocks) should be
applied to the block.

If there is an `else` component, then the closing brace, `else`, any following
clause, and the opening brace should all be on the same line. There should be a
single space before and after the `else` keyword. For example:

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

If the control line needs to be broken, then prefer to break before the `=` in
`* let` expressions and before `in` in a `for` expression; the following line
should be block indented. If the control line is broken for any reason, then the
opening brace should be on its own line and not indented. Examples:

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

Where the initial clause is multi-lined and ends with one or more closing
parentheses, square brackets, or braces, and there is nothing else on that line,
and that line is not indented beyond the indent on the first line of the control
flow expression, then the opening brace of the block should be put on the same
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

Formatters may place an `if else` or `if let else` on a single line if it occurs
in expression context (i.e., is not a standalone statement), it contains a
single `else` clause, and is *small*. For example:

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

Prefer not to line-break inside the discriminant expression. There must always
be a line break after the opening brace and before the closing brace. The match
arms must be block indented once:

```rust
match foo {
    // arms
}

let x = match foo.bar.baz() {
    // arms
};
```

Use a trailing comma for a match arm if and only if not using a block.

Never start a match arm pattern with `|`, e.g.,

```rust
match foo {
    // Don't do this.
    | foo => bar,
    // Or this.
    | a_very_long_pattern
    | another_pattern
    | yet_another_pattern
    | a_forth_pattern => {
        ...
    }
}
```

Prefer


```rust
match foo {
    foo => bar,
    a_very_long_pattern
    | another_pattern
    | yet_another_pattern
    | a_forth_pattern => {
        ...
    }
}
```

Avoid splitting the left-hand side (before the `=>`) of a match arm where
possible. If the right-hand side of the match arm is kept on the same line,
never use a block (unless the block is empty).

If the right-hand side consists of multiple statements or has line comments or
the start of the line cannot be fit on the same line as the left-hand side, use
a block.

The body of a block arm should be block indented once.

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
expression, then it may be started on the same line as the right-hand side. If
not, then it must be in a block. Example,

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

Where it is possible to use a block form on the right-hand side and avoid
breaking the left-hand side, do that. E.g.

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
clause, then you must use the above form:

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
not put the `if` clause on a newline. E.g.,

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
one line, then the pattern may be formatted across multiple lines with as many
clauses per line as possible. Again break before a `|`:

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
across multiple-lines, the outer call may be formatted as if it were a single-
line call. The same combining behaviour may be applied to any similar
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

Such behaviour should extend recursively, however, tools may choose to limit the
depth of nesting.

Only where the multi-line sub-expression is a closure with an explicit block,
this combining behaviour may be used where there are other arguments, as long as
all the arguments and the first line of the closure fit on the first line, the
closure is the last argument, and there is only one closure argument:

```rust
foo(first_arg, x, |param| {
    action();
    foo(param)
})
```


### Ranges

Do not put spaces in ranges, e.g., `0..10`, `x..=y`, `..x.len()`, `foo..`.

When writing a range with both upper and lower bounds, if the line must be
broken, break before the range operator and block indent the second line:

```rust
a_long_expression
    ..another_long_expression
```

For the sake of indicating precedence, we recommend that if either bound is a
compound expression, then use parentheses around it, e.g., `..(x + 1)`,
`(x.f)..(x.f.len())`, or `0..(x - 10)`.


### Hexadecimal literals

Hexadecimal literals may use upper- or lower-case letters, but they must not be
mixed within the same literal. Projects should use the same case for all
literals, but we do not make a recommendation for either lower- or upper-case.
Tools should have an option to convert mixed case literals to upper-case, and
may have an option to convert all literals to either lower- or upper-case.


## Patterns

Patterns should be formatted like their corresponding expressions. See the
section on `match` for additional formatting for patterns in match arms.
