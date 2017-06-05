- Feature Name: loop_break_value
- Start Date: 2016-05-20
- RFC PR: https://github.com/rust-lang/rfcs/pull/1624
- Rust Issue: https://github.com/rust-lang/rust/issues/37339

# Summary
[summary]: #summary

(This is a result of discussion of
[issue #961](https://github.com/rust-lang/rfcs/issues/961) and related to RFCs
[352](https://github.com/rust-lang/rfcs/pull/352) and
[955](https://github.com/rust-lang/rfcs/pull/955).)

Let a `loop { ... }` expression return a value via `break my_value;`.

# Motivation
[motivation]: #motivation

> Rust is an expression-oriented language. Currently loop constructs don't
> provide any useful value as expressions, they are run only for their
> side-effects. But there clearly is a "natural-looking", practical case,
> described in [this thread](https://github.com/rust-lang/rfcs/issues/961)
> and [this] RFC, where the loop expressions could have
> meaningful values. I feel that not allowing that case runs against the
> expression-oriented conciseness of Rust.
> [comment by golddranks](https://github.com/rust-lang/rfcs/issues/961#issuecomment-220820787)

Some examples which can be much more concisely written with this RFC:

```rust
// without loop-break-value:
let x = {
    let temp_bar;
    loop {
        ...
        if ... {
            temp_bar = bar;
            break;
        }
    }
    foo(temp_bar)
};

// with loop-break-value:
let x = foo(loop {
        ...
        if ... { break bar; }
    });

// without loop-break-value:
let computation = {
    let result;
    loop {
        if let Some(r) = self.do_something() {
            result = r;
            break;
        }
    }
    result.do_computation()
};
self.use(computation);

// with loop-break-value:
let computation = loop {
        if let Some(r) = self.do_something() {
            break r;
        }
    }.do_computation();
self.use(computation);
```

# Detailed design
[design]: #detailed-design

This proposal does two things: let `break` take a value, and let `loop` have a
result type other than `()`.

### Break Syntax

Four forms of `break` will be supported:

1.  `break;`
2.  `break 'label;`
3.  `break EXPR;`
4.  `break 'label EXPR;`

where `'label` is the name of a loop and `EXPR` is an expression. `break` and `break 'label` become
equivalent to `break ()` and `break 'label ()` respectively.

### Result type of loop

Currently the result type of a 'loop' without 'break' is `!` (never returns),
which may be coerced to any type. The result type of a 'loop' with a 'break'
is `()`. This is important since a loop may appear as the last expression of
a function:

```rust
fn f() {
    loop {
        do_something();
        // never breaks
    }
}
fn g() -> () {
    loop {
        do_something();
        if Q() { break; }
    }
}
fn h() -> ! {
    loop {
        do_something();
        // this loop must diverge for the function to typecheck
    }
}
```

This proposal allows 'loop' expression to be of any type `T`, following the same typing and
inference rules that are applicable to other expressions in the language. Type of `EXPR` in every
`break EXPR` and `break 'label EXPR` must be coercible to the type of the loop the `EXPR` appears
in.

<!-- [ASIDE] The above paragraph captures pretty much every important typesystem interaction:
     * `!` coerces to any type, so `break (expr: !)` works regardless of the loop type;
         * It also does not preclude having `loop { break (expr: !) }: !`
     * It, works well for `T = !`, because nothing in this paragraph demands to have breaks, only
       sets requirements type coercibility;
     * Similarly it works well for `T = ()` with all break forms because of `break` ≡ `break ()`.
     * Finally the `loop { ... }: S` also works fine, because it requires every EXPR to coerce to
       S, which is consistent with the behaviour of `if` bodies, `match` bodies etc.
     * It also retains the `break`-less loop may be of type `!` property, because there’s no EXPRs
       that have to coerce to `!`, whereas if it contains some `break`, then () cannot coerce to !.
-->

It is an error if these types do not agree or if the compiler's type deduction rules do not yield a
concrete type.

Examples of errors:

```rust
// error: loop type must be () and must be i32
let a: i32 = loop { break; };
// error: loop type must be i32 and must be &str
let b: i32 = loop { break "I am not an integer."; };
// error: loop type must be Option<_> and must be &str
let c = loop {
    if Q() {
        break "answer";
    } else {
        break None;
    }
};
fn z() -> ! {
    // function does not return
    // error: loop may break (same behaviour as before)
    loop {
        if Q() { break; }
    }
}
```

Example showing the equivalence of `break;` and `break ();`:

```rust
fn y() -> () {
    loop {
        if coin_flip() {
            break;
        } else {
            break ();
        }
    }
}
```

Coercion examples:

```rust
// ! coerces to any type
loop {}: ();
loop {}: u32;
loop {
    break (loop {}: !);
}: u32;
loop {
    // ...
    break 42;
    // ...
    break panic!();
}: u32;

// break EXPRs are not of the same type, but both coerce to `&[u8]`.
let x = [0; 32];
let y = [0; 48];
loop {
    // ...
    break &x;
    // ...
    break &y;
}: &[u8];
```


### Result value

A loop only yields a value if broken via some form of `break ...;` statement,
in which case it yields the value resulting from the evaulation of the
statement's expression (`EXPR` above), or `()` if there is no `EXPR`
expression.

Examples:

```rust
assert_eq!(loop { break; }, ());
assert_eq!(loop { break 5; }, 5);
let x = 'a loop {
    'b loop {
        break 'a 1;
    }
    break 'a 2;
};
assert_eq!(x, 1);
```

# Drawbacks
[drawbacks]: #drawbacks

The proposal changes the syntax of `break` statements, requiring updates to
parsers and possibly syntax highlighters.

# Alternatives
[alternatives]: #alternatives

No alternatives to the design have been suggested. It has been suggested that
the feature itself is unnecessary, and indeed much Rust code already exists
without it, however the pattern solves some cases which are difficult to handle
otherwise and allows more flexibility in code layout.

# Unresolved questions
[unresolved]: #unresolved-questions

### Extension to for, while, while let

A frequently discussed issue is extension of this concept to allow `for`,
`while` and `while let` expressions to return values in a similar way. There is
however a complication: these expressions may also terminate "naturally" (not
via break), and no consensus has been reached on how the result value should
be determined in this case, or even the result type.

There are three options:

1.  Do not adjust `for`, `while` or `while let` at this time
2.  Adjust these control structures to return an `Option<T>`, returning `None`
    in the default case
3.  Specify the default return value via some extra syntax

#### Via `Option<T>`

Unfortunately, option (2) is not possible to implement cleanly without breaking
a lot of existing code: many functions use one of these control structures in
tail position, where the current "value" of the expression, `()`, is implicitly
used:

```rust
// function returns `()`
fn print_my_values(v: &Vec<i32>) {
    for x in v {
        println!("Value: {}", x);
    }
    // loop exits with `()` which is implicitly "returned" from the function
}
```

Two variations of option (2) are possible:

*   Only adjust the control structures where they contain a `break EXPR;` or
    `break 'label EXPR;` statement. This may work but would necessitate that
    `break;` and `break ();` mean different things.
*   As a special case, make `break ();` return `()` instead of `Some(())`,
    while for other values `break x;` returns `Some(x)`.

#### Via extra syntax for the default value

Several syntaxes have been proposed for how a control structure's default value
is set. For example:

```rust
fn first<T: Copy>(list: Iterator<T>) -> Option<T> {
    for x in list {
        break Some(x);
    } else default {
        None
    }
}
```

or:

```rust
let x = for thing in things default "nope" {
    if thing.valid() { break "found it!"; }
}
```

There are two things to bear in mind when considering new syntax:

*   It is undesirable to add a new keyword to the list of Rust's keywords
*   It is strongly desirable that unbounded lookahead is *not* required while syntax
    parsing Rust code

For more discussion on this topic, see [issue #961](https://github.com/rust-lang/rfcs/issues/961).
