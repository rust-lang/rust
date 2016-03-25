- Feature Name: closure_to_fn_coercion
- Start Date: 2016-03-25
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

A non-capturing (that is, does not `Clone` or `move` any local variables) should be
coercable to a function pointer (`fn`).

# Motivation
[motivation]: #motivation

Currently in rust, it is impossible to bind anything but a pre-defined function
as a function pointer. When dealing with closures, one must either rely upon
rust's type-inference capabilities, or use the `Fn` trait to abstract for any
closure with a certain type signature.

What is not possible, though, is to define a function while at the same time
binding it to a function pointer.

This is mainly used for convenience purposes, but in certain situations
the lack of ability to do so creates a significant amount of boilerplate code.
For example, when attempting to create an array of small, simple, but unique functions,
it would be necessary to pre-define each and every function beforehand:

```rust
fn inc_0(var: &mut u32) {}
fn inc_1(var: &mut u32) { *var += 1; }
fn inc_2(var: &mut u32) { *var += 2; }
fn inc_3(var: &mut u32) { *var += 3; }

const foo: [fn(&mut u32); 4] = [
  inc_0,
  inc_1,
  inc_2,
  inc_3,
];
```

This is a trivial example, and one that might not seem too consequential, but the
code doubles with every new item added to the array. With very many elements,
the duplication begins to seem unwarranted.

Another option, of course, is to use an array of `Fn` instead of `fn`:

```rust
const foo: [&'static Fn(&mut u32); 4] = [
  &|var: &mut u32| {},
  &|var: &mut u32| *var += 1,
  &|var: &mut u32| *var += 2,
  &|var: &mut u32| *var += 3,
];
```

And this seems to fix the problem. Unfortunately, however, looking closely one
can see that because we use the `Fn` trait, an extra layer of indirection
is added when attempting to run `foo[n](&mut bar)`.

Rust must use dynamic dispatch because a closure is secretly a struct that
contains references to captured variables, and the code within that closure
must be able to access those references stored in the struct.

In the above example, though, no variables are captured by the closures,
so in theory nothing would stop the compiler from treating them as anonymous
functions. By doing so, unnecessary indirection would be avoided. In situations
where this function pointer array is particularly hot code, the optimization
would be appreciated.

# Detailed design
[design]: #detailed-design

In C++, non-capturing lambdas (the C++ equivalent of closures) "decay" into function pointers
when they do not need to capture any variables. This is used, for example, to pass a lambda
into a C function:

```cpp
void foo(void (*foobar)(void)) {
    // impl
}
void bar() {
    foo([]() { /* do something */ });
}
```

With this proposal, rust users would be able to do the same:

```rust
fn foo(foobar: fn()) {
    // impl
}
fn bar() {
    foo(|| { /* do something */ });
}
```

Using the examples within ["Motivation"](#motivation), the code array would
be simplified to no performance detriment:

```rust
const foo: [fn(&mut u32); 4] = [
  |var: &mut u32| {},
  |var: &mut u32| *var += 1,
  |var: &mut u32| *var += 2,
  |var: &mut u32| *var += 3,
];
```

# Drawbacks
[drawbacks]: #drawbacks

To a rust user, there is no drawback to this new coercion from closures to `fn` types.

The only drawback is that it would add some amount of complexity to the type system.

# Alternatives
[alternatives]: #alternatives

## Anonymous function syntax

With this alternative, rust users would be able to directly bind a function
to a variable, without needing to give the function a name.

```rust
let foo = fn() { /* do something */ };
foo();
```

```rust
const foo: [fn(&mut u32); 4] = [
  fn(var: &mut u32) {},
  fn(var: &mut u32) { *var += 1 },
  fn(var: &mut u32) { *var += 2 },
  fn(var: &mut u32) { *var += 3 },
];
```

This isn't ideal, however, because it would require giving new semantics
to `fn` syntax.

## Aggressive optimization

This is possibly unrealistic, but an alternative would be to continue encouraging
the use of closures with the `Fn` trait, but conduct heavy optimization to determine
when the used closure is "trivial" and does not need indirection.

Of course, this would probably significantly complicate the optimization process, and
would have the detriment of not being easily verifiable by the programmer without
checking the disassembly of their program.

# Unresolved questions
[unresolved]: #unresolved-questions

None
