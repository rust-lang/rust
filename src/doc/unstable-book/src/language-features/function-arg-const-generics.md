# function_arg_const_generics

The tracking issue for this feature is: [#163261]

[#163261]: https://github.com/rust-lang/rust/issues/163261

------------------------

The `#![feature(function_arg_const_generics)]` allows const generic parameters to be declared in a function's value parameter list and supplied as ordinary call arguments, rather than through generic arguments (turbofish syntax).

Warning: This feature is incomplete; its design and syntax may change.

This feature requires either `#![feature(gca_min_const_items)]` or `#![feature(gca_adts)]`.

## Examples

Normally, const generic parameters are declared in the generic parameter list and supplied using turbofish syntax:

```rust
fn foo<const N: usize>(value: u8) {
    let _ = N;
}

fn main() {
    foo::<42>(10);
}
```

With `function_arg_const_generics`, the const generic parameter can instead be declared in the function's value parameter list:

```rust
#![allow(incomplete_features)]
#![feature(function_arg_const_generics, gca_min_const_items)]

fn foo(const N: usize, value: u8) {
    let _ = (N, value);
}

fn main() {
    foo(42, 10);
}
```

Here, `N` is still a const generic parameter, but its value is supplied as the first argument to the function call.

### Methods and trait functions

Argument position const generic parameters are supported in free functions, inherent methods, trait methods, and trait implementations.

```rust
#![allow(incomplete_features)]
#![feature(function_arg_const_generics, gca_min_const_items)]

struct Foo;

impl Foo {
    fn bar(&self, const N: usize) {
        let _ = N;
    }
}

trait Trait {
    fn baz(const N: usize);
}

impl Trait for Foo {
    fn baz(const N: usize) {
        let _ = N;
    }
}

fn main() {
    Foo.bar(10);
    Foo::baz(20);
}
```

### Supported const arguments

Currently, only the following forms of const arguments are supported:

- Literals.
- Paths to const generic parameters.
- Paths to const items, provided the const item is marked for use in the type system, for example using `gca!`.

```rust
#![allow(incomplete_features)]
#![feature(function_arg_const_generics, gca_min_const_items)]

fn foo(const N: usize) {
    let _ = N;
}

const VALUE: usize = core::gca!(42);

fn bar<const K: usize>() {
    foo(K);
}

fn main() {
    foo(10);    // OK: integer literal
    foo(VALUE); // OK: const item marked with gca!
    bar::<20>();
}
```

### Restrictions

Argument position const generic parameters have the following restrictions:

- They must be declared using a plain identifier, such as `const N: usize`.
- They cannot be explicitly supplied using turbofish syntax.
- Arguments must use one of the supported const argument forms described above.

For example, a runtime local variable cannot be used as a const argument:

```rust,compile_fail
#![allow(incomplete_features)]
#![feature(function_arg_const_generics, gca_min_const_items)]

fn foo(const N: usize) {}

fn main() {
    let n = 10;
    foo(n); // ERROR: runtime local variable
}
```

Similarly, argument position const generic parameters cannot be specified using turbofish syntax:

```rust,compile_fail,E0107
#![allow(incomplete_features)]
#![feature(function_arg_const_generics, gca_min_const_items)]

fn foo(const N: usize) {}

fn main() {
    foo::<10>(10); // ERROR: N cannot be specified using turbofish
}
```

Argument position const generic parameters are not currently supported in function pointer types, `Fn()` trait syntax, or foreign function declarations.
