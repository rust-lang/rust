- Feature Name: label_break_value
- Start Date: 2017-06-26
- RFC PR: [#2046](https://github.com/rust-lang/rfcs/pull/2046)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Allow a break not only out of `loop`, but of labelled blocks with no loop. Like `loop`, this break can carry a value.

This depends on [RFC 1624](https://github.com/rust-lang/rfcs/blob/master/text/1624-loop-break-value.md) landing. I proposed this [here](https://github.com/rust-lang/rust/issues/37339#issuecomment-306573033). An identical proposal was part of the [explanation for trait based exception handling](https://github.com/rust-lang/rfcs/blob/master/text/0243-trait-based-exception-handling.md#early-exit-from-any-block).

# Motivation
[motivation]: #motivation

In its simplest form, this allows you to terminate a block early, the same way that `return` allows you to terminate a function early.

```
'block: {
    do_thing();
    if condition_not_met() {
        break 'block;
    }
    do_next_thing();
    if condition_not_met() {
        break 'block;
    }
    do_last_thing();
}
```
Following RFC 1624, this, like `return`, can also carry a value:
```
let result = 'block: {
    if foo() { break 'block 1; }
    if bar() { break 'block 2; }
    3
};
```
RFC 1624 opted not to allow options to be returned from `for` or `while` loops, since no good option could be found for the syntax, and it was hard to do it in a natural way. This proposal gives us a natural way to handle such loops with no changes to their syntax:
```
let result = 'block: {
    for v in container.iter() {
        if v > 0 { break 'block v; }
    }
    0
};
```
This extension handles searches more complex than loops in the same way:
```
let result = 'block: {
    for v in first_container.iter() {
        if v > 0 { break 'block v; }
    }
    for v in second_container.iter() {
        if v > 0 { break 'block v; }
    }
    0
};
```
# Detailed design
[design]: #detailed-design
```
'BLOCK_LABEL: { EXPR }
```
would simply be syntactic sugar for
```
'BLOCK_LABEL: loop { break { EXPR } }
```
except that unlabelled `break`s or `continue`s which would bind to the implicit `loop` are forbidden inside the *EXPR*.

This is perhaps not a conceptually simpler thing, but it has the advantage that all of the wrinkles are already well understood as a result of the work that went into RFC 1624. If *EXPR* contains explicit `break` statements as well as the implicit one, the compiler must be able to infer a single concrete type from the expressions in all of these `break` statements, including the whole of *EXPR*; this concrete type will be the type of the expression that the labelled block represents.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

This can be taught alongside loop-based examples of labelled breaks.

# Drawbacks
[drawbacks]: #drawbacks

The proposal adds new syntax to blocks, requiring updates to parsers and possibly syntax highlighters.

# Alternatives
[alternatives]: #alternatives

This feature isn't necessary; however in my own code, I often find myself breaking something out into a function simply in order to return early, and the accompanying verbosity of passing types and return values is often not worth it. 

Another alternative would be to revisit one of the proposals to add syntax to `for` and `while`.

We have three options for handling an unlabelled `break` or `continue` inside a labelled block:

 - compile error on both `break` and `continue`
 - bind `break` to the labelled block, compile error on `continue`
 - bind `break` and `continue` through the labelled block to a containing `loop`/`while`/`for`

This RFC chooses the first option since it's the most conservative, in that it would be possible to switch to a different behaviour later without breaking working programs. The second is the simplest, but makes a large difference between labelled and unlabelled blocks, and means that a label might be used even when it's never referred to. The third is consistent with unlabelled blocks and with Java, but seems like a rich potential source of confusion.

# Unresolved questions
[unresolved]: #unresolved-questions

None outstanding that I know about.
