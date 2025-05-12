# Tracking moves and initialization

Part of the borrow checker's job is to track which variables are
"initialized" at any given point in time -- this also requires
figuring out where moves occur and tracking those.

## Initialization and moves

From a user's perspective, initialization -- giving a variable some
value -- and moves -- transferring ownership to another place -- might
seem like distinct topics. Indeed, our borrow checker error messages
often talk about them differently. But **within the borrow checker**,
they are not nearly as separate. Roughly speaking, the borrow checker
tracks the set of "initialized places" at any point in the source
code. Assigning to a previously uninitialized local variable adds it
to that set; moving from a local variable removes it from that set.

Consider this example:

```rust,ignore
fn foo() {
    let a: Vec<u32>;
    
    // a is not initialized yet
    
    a = vec![22];
    
    // a is initialized here
    
    std::mem::drop(a); // a is moved here
    
    // a is no longer initialized here

    let l = a.len(); //~ ERROR
}
```

Here you can see that `a` starts off as uninitialized; once it is
assigned, it becomes initialized. But when `drop(a)` is called, that
moves `a` into the call, and hence it becomes uninitialized again.

## Subsections

To make it easier to peruse, this section is broken into a number of
subsections:

- [Move paths](./moves_and_initialization/move_paths.html) the
  *move path* concept that we use to track which local variables (or parts of
  local variables, in some cases) are initialized.
- TODO *Rest not yet written* =)
