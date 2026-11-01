# `loop_match`

The tracking issue for this feature is: [#132306]

[#132306]: https://github.com/rust-lang/rust/issues/132306

------

The `#[loop_match]` and `#[const_continue]` attributes can be used to improve the code
generation of logic that fits this shape:

```ignore (pseudo-rust)
loop {
    state = 'blk: {
        match state {
            State::A => {
                break 'blk State::B
            }
            State::B => { /* ... */ }
            /* ... */
        }
    }
}
```

Here the loop itself can be annotated with `#[loop_match]`, and any `break 'blk` with
`#[const_continue]` if the value is know at compile time:

```ignore (pseudo-rust)
#[loop_match]
loop {
    state = 'blk: {
        match state {
            State::A => {
                #[const_continue]
                break 'blk State::B
            }
            State::B => { /* ... */ }
            /* ... */
        }
    }
}
```

The observable behavior of this loop is exactly the same as without the extra attributes.
The difference is in the generated output: normally, when the state is `A`, control flow
moves from the `A` branch, back to the top of the loop, then to the `B` branch. With the
attributes, The `A` branch will immediately jump to the `B` branch.

Removing the indirection can be beneficial for stack usage and branch prediction, and
enables other optimizations by clearly splitting out the control flow paths that your
program will actually use.
