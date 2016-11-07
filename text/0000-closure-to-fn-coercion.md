- Feature Name: closure_to_fn_coercion
- Start Date: 2016-03-25
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

A non-capturing (that is, does not `Clone` or `move` any local variables) closure
should be coercable to a function pointer (`fn`).

# Motivation
[motivation]: #motivation

Currently in Rust, it is impossible to bind anything but a pre-defined function
as a function pointer. When dealing with closures, one must either rely upon
Rust's type-inference capabilities, or use the `Fn` trait to abstract for any
closure with a certain type signature.

It is not possible to define a function while at the same time binding it to a
function pointer.

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

Note that once explicitly assigned to an `Fn` trait, the closure can no longer be
coerced into `fn`, even if it has no captures. Just as we cannot do:

```rust
let a: u32 = 0; // Coercion
let b: i32 = a; // Can't re-coerce
let x: *const u32 = &a; // Coercion
let y: &u32 = x; // Can't re-coerce
```

We can't similarly re-coerce a `Fn` trait.
```rust
let a: &Fn(u32) -> u32 = |foo: u32| { foo + 1 };
let b: fn(u32) -> u32 = *a; // Can't re-coerce
```

# Drawbacks
[drawbacks]: #drawbacks

This proposal could potentially allow Rust users to accidentally constrain their APIs.
In the case of a crate, a user accidentally returning `fn` instead of `Fn` may find
that their code compiles at first, but breaks when the user later needs to capture variables:

```rust
// The specific syntax is more convenient to use
fn func_specific(&self) -> (fn() -> u32) {
  || return 0
}

fn func_general<'a>(&'a self) -> impl Fn() -> u32 {
  move || return self.field
}
```

In the above example, the API author could start off with the specific version of the function,
and by circumstance later need to capture a variable. The required change from `fn` to `Fn` could
be a breaking change.

We do expect crate authors to measure their API's flexibility in other areas, however, as when
determining whether to take `&self` or `&mut self`. Taking a similar situation to the above: 

```rust
fn func_specific<'a>(&'a self) -> impl Fn() -> u32 {
  move || return self.field
}
    
fn func_general<'a>(&'a mut self) -> impl FnMut() -> u32 {
  move || { self.field += 1; return self.field; }
}
```

This drawback is probably outweighed by convenience, simplicity, and the potential for optimization
that comes with the proposed changes, however.

# Alternatives
[alternatives]: #alternatives

## Anonymous function syntax

With this alternative, Rust users would be able to directly bind a function
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
