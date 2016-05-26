- Feature Name: loop_break_value
- Start Date: 2016-05-20
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

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

where `'label` is the name of a loop and `EXPR` is an expression.

### Result type of loop

Currently the result type of a 'loop' without 'break' is `!` (never returns),
which may be coerced to any type), and the result type of a 'loop' with 'break'
is `()`. This is important since a loop may appear as
the last expression of a function:

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
        // this loop is not allowed to break due to inferred `!` type
    }
}
```

This proposal changes the result type of 'loop' to `T`, where:

*   if a loop is "broken" via `break;` or `break 'label;`, the loop's result type must be `()`
*   if a loop is "broken" via `break EXPR;` or `break 'label EXPR;`, `EXPR` must evaluate to type `T`
*   as a special case, if a loop is "broken" via `break EXPR;` or `break 'label EXPR;` where `EXPR` evaluates to type `!` (does not return), this does not place a constraint on the type of the loop
*   if external constaint on the loop's result type exist (e.g. `let x: S = loop { ... };`), then `T` must be coercible to this type

It is an error if these types do not agree or if the compiler's type deduction
rules do not yield a concrete type.

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

Examples involving `!`:

```rust
fn f() -> () {
    // ! coerces to ()
    loop {}
}
fn g() -> u32 {
    // ! coerces to u32
    loop {}
}
fn z() -> ! {
    loop {
        break panic!();
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

It would be possible to allow `for`, `while` and `while let` expressions return
values in a similar way; however, these expressions may also terminate
"naturally" (not via break), and no consensus has been reached on how the
result value should be determined in this case, or even the result type.
It is thus proposed not to change these expressions at this time.

It should be noted that `for`, `while` and `while let` can all be emulated via
`loop`, so perhaps allowing the former to return values is less important.
Alternatively, a keyword such as `else default` could be used to
specify the other exit value as in:

```rust
fn first<T: Copy>(list: Iterator<T>) -> Option<T> {
    for x in list {
        break Some(x);
    } else default {
        None
    }
}
```

The exact syntax is disputed; (JelteF has some suggestions which should work
without infinite parser lookahead)
[https://github.com/rust-lang/rfcs/issues/961#issuecomment-220728894].
It is suggested that this RFC should not be blocked on this issue since
loop-break-value can still be implemented in the manner above after this RFC.
