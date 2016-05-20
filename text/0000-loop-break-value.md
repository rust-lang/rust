- Feature Name: loop_break_value
- Start Date: 2016-05-20
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

(This is a result of discussion of issue #961 and related to RFCs
[352](https://github.com/rust-lang/rfcs/pull/352) and
[955](https://github.com/rust-lang/rfcs/pull/955).)

Let a `loop { ... }` expression return a value via `break my_value;`.

# Motivation
[motivation]: #motivation

This pattern is currently hard to implement without resorting to a function or
closure wrapping the loop:

```rust
fn f() {
    let outcome = loop {
        // get and process some input, e.g. from the user or from a list of
        // files
        let result = get_result();
        
        if successful() {
            break result;
        }
        // otherwise keep trying
    };
    
    use_the_result(outcome);
}
```

In some cases, one can simply move `use_the_result(outcome)` into the loop, but
sometimes this is undesirable and sometimes impossible due to lifetimes.

# Detailed design
[design]: #detailed-design

This proposal does two things: let `break` take a value, and let `loop` have a
result type other than `()`.

### break syntax

Four forms of `break` will be supported:

1.  `break;`
2.  `break 'label;`
3.  `break EXPR;`
4.  `break 'label EXPR;`

where `'label` is the name of a looping construct and `EXPR` is an evaluable
expression.

### result type

Currently the result-type of a 'loop' without 'break' is `!` (never returns),
which may be coerced to `()`. This is important since a loop may appear as
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
        // this loop is not allowed to break due to `!` result type
    }
}
```

This proposal changes the result type to `T`, where:

*   a loop which is never "broken" via `break` has result-type `!` (coercible to `()`)
*   a loop's return type may be deduced from its context, e.g. `let x: T = loop { ... };`
*   where a loop is "broken" via `break;` or `break 'label;`, its result type is `()`
*   where a loop is "broken" via `break EXPR;` or `break 'label EXPR;`, `EXPR` must evaluate to type `T`

It is an error if these types do not agree. Examples:

```rust
// error: loop type must be ! or () not i32
let a: i32 = loop {};
// error: loop type must be i32 and must be &str
let b: i32 = loop { break "I am not an integer."; };
// error: loop type must be Option<_> and must be &str
let c = loop {
    if Q() {
        "answer"
    } else {
        None
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

### result value

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

The type of `loop` expressions is no longer fixed and cannot be explicitly
typed.

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

See discussion of #961 for more on this topic.
