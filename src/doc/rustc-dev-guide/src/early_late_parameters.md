
# Early vs Late bound parameters

<!-- toc -->

> **NOTE**: This chapter largely talks about early/late bound as being solely relevant when discussing function item types/function definitions. This is potentially not completely true, async blocks and closures should likely be discussed somewhat in this chapter.

## What does it mean to be "early" bound or "late" bound

Every function definition has a corresponding ZST that implements the `Fn*` traits known as a [function item type][function_item_type]. This part of the chapter will talk a little bit about the "desugaring" of function item types as it is useful context for explaining the difference between early bound and late bound generic parameters.

Let's start with a very trivial example involving no generic parameters:

```rust
fn foo(a: String) -> u8 {
    # 1
    /* snip */
}
```

If we explicitly wrote out the definitions for the function item type corresponding to `foo` and its associated `Fn` impl it would look something like this:
```rust,ignore
struct FooFnItem;

impl Fn<(String,)> for FooFnItem {
    type Output = u8;
    /* fn call(&self, ...) -> ... { ... } */
}
```

The builtin impls for the `FnMut`/`FnOnce` traits as well as the impls for `Copy` and `Clone` were omitted for brevity reasons (although these traits *are* implemented for function item types).

A slightly more complicated example would involve introducing generic parameters to the function:
```rust
fn foo<T: Sized>(a: T) -> T { 
    # a
    /* snip */
}
```
Writing out the definitions would look something like this:
```rust,ignore
struct FooFnItem<T: Sized>(PhantomData<fn(T) -> T>);

impl<T: Sized> Fn<(T,)> for FooFnItem<T> {
    type Output = T;
    /* fn call(&self, ...) -> ... { ... } */
}
```

Note that the function item type `FooFnItem` is generic over some type parameter `T` as defined on the function `foo`. However, not all generic parameters defined on functions are also defined on the function item type as demonstrated here:
```rust
fn foo<'a, T: Sized>(a: &'a T) -> &'a T {
    # a
    /* snip */
}
```
With its "desugared" form looking like so:
```rust,ignore
struct FooFnItem<T: Sized>(PhantomData<for<'a> fn(&'a T) -> &'a T>);

impl<'a, T: Sized> Fn<(&'a T,)> for FooFnItem<T> {
    type Output = &'a T;
    /* fn call(&self, ...) -> ... { ... } */
}
```

The lifetime parameter `'a` from the function `foo` is not present on the function item type `FooFnItem` and is instead introduced on the builtin impl solely for use in representing the argument types.

Generic parameters not all being defined on the function item type means that there are two steps where generic arguments are provided when calling a function.
1. Naming the function (e.g. `let a = foo;`) the arguments for `FooFnItem` are provided. 
2. Calling the function (e.g. `a(&10);`) any parameters defined on the builtin impl are provided.

This two-step system is where the early vs late naming scheme comes from, early bound parameters are provided in the *earliest* step (naming the function), whereas late bound parameters are provided in the *latest* step (calling the function). 

Looking at the desugaring from the previous example we can tell that `T` is an early bound type parameter and `'a` is a late bound lifetime parameter as `T` is present on the function item type but `'a` is not. See this example of calling `foo` annotated with where each generic parameter has an argument provided:
```rust
fn foo<'a, T: Sized>(a: &'a T) -> &'a T {
    # a
    /* snip */
}

// Here we provide a type argument `String` to the
// type parameter `T` on the function item type
let my_func = foo::<String>;

// Here (implicitly) a lifetime argument is provided
// to the lifetime parameter `'a` on the builtin impl.
my_func(&String::new());
```

[function_item_type]: https://doc.rust-lang.org/reference/types/function-item.html

## Differences between early and late bound parameters

### Higher ranked function pointers and trait bounds 

A generic parameter being late bound allows for more flexible usage of the function item. For example if we have some function `foo` with an early bound lifetime parameter and some function `bar` with a late bound lifetime parameter `'a` we would have the following builtin `Fn` impls:
```rust,ignore
impl<'a> Fn<(&'a String,)> for FooFnItem<'a> { /* ... */ }
impl<'a> Fn<(&'a String,)> for BarFnItem { /* ... */ }
```

The `bar` function has a strictly more flexible signature as the function item type can be called with a borrow with *any* lifetime, whereas the `foo` function item type would only be callable with a borrow with the same lifetime on the function item type. We can show this by simply trying to call `foo`'s function item type multiple times with different lifetimes:

```rust
// The `'a: 'a` bound forces this lifetime to be early bound.
fn foo<'a: 'a>(b: &'a String) -> &'a String { b }
fn bar<'a>(b: &'a String) -> &'a String { b }

// Early bound generic parameters are instantiated here when naming
// the function `foo`. As `'a` is early bound an argument is provided.
let f = foo::<'_>;

// Both function arguments are required to have the same lifetime as
// the lifetime parameter being early bound means that `f` is only
// callable for one specific lifetime.
//
// As we call this with borrows of different lifetimes, the borrow checker
// will error here.
f(&String::new());
f(&String::new());
```

In this example we call `foo`'s function item type twice, each time with a borrow of a temporary. These two borrows could not possible have lifetimes that overlap as the temporaries are only alive during the function call, not after. The lifetime parameter on `foo` being early bound requires all callers of `f` to provide a borrow with the same lifetime, as this is not possible the borrow checker errors.

If the lifetime parameter on `foo` was late bound this would be able to compile as each caller could provide a different lifetime argument for its borrow. See the following example which demonstrates this using the `bar` function defined above:

```rust
# fn foo<'a: 'a>(b: &'a String) -> &'a String { b }
# fn bar<'a>(b: &'a String) -> &'a String { b }
#
// Early bound parameters are instantiated here, however as `'a` is
// late bound it is not provided here.
let b = bar;

// Late bound parameters are instantiated separately at each call site
// allowing different lifetimes to be used by each caller.
b(&String::new());
b(&String::new());
```

This is reflected in the ability to coerce function item types to higher ranked function pointers and prove higher ranked `Fn` trait bounds. We can demonstrate this with the following example:
```rust
// The `'a: 'a` bound forces this lifetime to be early bound.
fn foo<'a: 'a>(b: &'a String) -> &'a String { b }
fn bar<'a>(b: &'a String) -> &'a String { b }

fn accepts_hr_fn(_: impl for<'a> Fn(&'a String) -> &'a String) {}

fn higher_ranked_trait_bound() {
    let bar_fn_item = bar;
    accepts_hr_fn(bar_fn_item);

    let foo_fn_item = foo::<'_>;
    // errors
    accepts_hr_fn(foo_fn_item);
}

fn higher_ranked_fn_ptr() {
    let bar_fn_item = bar;
    let fn_ptr: for<'a> fn(&'a String) -> &'a String = bar_fn_item;
    
    let foo_fn_item = foo::<'_>;
    // errors
    let fn_ptr: for<'a> fn(&'a String) -> &'a String = foo_fn_item;
}
```

In both of these cases the borrow checker errors as it does not consider `foo_fn_item` to be callable with a borrow of any lifetime. This is due to the fact that the lifetime parameter on `foo` is early bound, causing `foo_fn_item` to have a type of `FooFnItem<'_>` which (as demonstrated by the desugared `Fn` impl) is only callable with a borrow of the same lifetime `'_`.

### Turbofishing in the presence of late bound parameters

As mentioned previously, the distinction between early and late bound parameters means that there are two places where generic parameters are instantiated:
- When naming a function (early)
- When calling a function (late)

There is currently no syntax for explicitly specifying generic arguments for late bound parameters during the call step; generic arguments can only be specified for early bound parameters when naming a function.
The syntax `foo::<'static>();`, despite being part of a function call, behaves as `(foo::<'static>)();` and instantiates the early bound generic parameters on the function item type.

See the following example:
```rust
fn foo<'a>(b: &'a u32) -> &'a u32 { b }

let f /* : FooFnItem<????> */ = foo::<'static>;
```

The above example errors as the lifetime parameter `'a` is late bound and so cannot be instantiated as part of the "naming a function" step. If we make the lifetime parameter early bound we will see this code start to compile:
```rust
fn foo<'a: 'a>(b: &'a u32) -> &'a u32 { b }

let f /* : FooFnItem<'static> */ = foo::<'static>;
```

What the current implementation of the compiler aims to do is error when specifying lifetime arguments to a function that has both early *and* late bound lifetime parameters. In practice, due to excessive breakage, some cases are actually only future compatibility warnings ([#42868](https://github.com/rust-lang/rust/issues/42868)):
- When the amount of lifetime arguments is the same as the number of early bound lifetime parameters a FCW is emitted instead of an error
- An error is always downgraded to a FCW when using method call syntax

To demonstrate this we can write out the different kinds of functions and give them both a late and early bound lifetime:
```rust,ignore
fn free_function<'a: 'a, 'b>(_: &'a (), _: &'b ()) {}

struct Foo;

trait Trait: Sized {
    fn trait_method<'a: 'a, 'b>(self, _: &'a (), _: &'b ());
    fn trait_function<'a: 'a, 'b>(_: &'a (), _: &'b ());
}

impl Trait for Foo {
    fn trait_method<'a: 'a, 'b>(self, _: &'a (), _: &'b ()) {}
    fn trait_function<'a: 'a, 'b>(_: &'a (), _: &'b ()) {}
}

impl Foo {
    fn inherent_method<'a: 'a, 'b>(self, _: &'a (), _: &'b ()) {}
    fn inherent_function<'a: 'a, 'b>(_: &'a (), _: &'b ()) {}
}
```

Then, for the first case, we can call each function with a single lifetime argument (corresponding to the one early bound lifetime parameter) and note that it only results in a FCW rather than a hard error.
```rust
#![deny(late_bound_lifetime_arguments)]

# fn free_function<'a: 'a, 'b>(_: &'a (), _: &'b ()) {}
#
# struct Foo;
#
# trait Trait: Sized {
#     fn trait_method<'a: 'a, 'b>(self, _: &'a (), _: &'b ());
#     fn trait_function<'a: 'a, 'b>(_: &'a (), _: &'b ());
# }
#
# impl Trait for Foo {
#     fn trait_method<'a: 'a, 'b>(self, _: &'a (), _: &'b ()) {}
#     fn trait_function<'a: 'a, 'b>(_: &'a (), _: &'b ()) {}
# }
#
# impl Foo {
#     fn inherent_method<'a: 'a, 'b>(self, _: &'a (), _: &'b ()) {}
#     fn inherent_function<'a: 'a, 'b>(_: &'a (), _: &'b ()) {}
# }
#
// Specifying as many arguments as there are early
// bound parameters is always a future compat warning
Foo.trait_method::<'static>(&(), &());
Foo::trait_method::<'static>(Foo, &(), &());
Foo::trait_function::<'static>(&(), &());
Foo.inherent_method::<'static>(&(), &());
Foo::inherent_function::<'static>(&(), &());
free_function::<'static>(&(), &());
```

For the second case we call each function with more lifetime arguments than there are lifetime parameters (be it early or late bound) and note that method calls result in a FCW as opposed to the free/associated functions which result in a hard error:
```rust
# fn free_function<'a: 'a, 'b>(_: &'a (), _: &'b ()) {}
#
# struct Foo;
#
# trait Trait: Sized {
#     fn trait_method<'a: 'a, 'b>(self, _: &'a (), _: &'b ());
#     fn trait_function<'a: 'a, 'b>(_: &'a (), _: &'b ());
# }
#
# impl Trait for Foo {
#     fn trait_method<'a: 'a, 'b>(self, _: &'a (), _: &'b ()) {}
#     fn trait_function<'a: 'a, 'b>(_: &'a (), _: &'b ()) {}
# }
#
# impl Foo {
#     fn inherent_method<'a: 'a, 'b>(self, _: &'a (), _: &'b ()) {}
#     fn inherent_function<'a: 'a, 'b>(_: &'a (), _: &'b ()) {}
# }
#
// Specifying more arguments than there are early
// bound parameters is a future compat warning when
// using method call syntax.
Foo.trait_method::<'static, 'static, 'static>(&(), &());
Foo.inherent_method::<'static, 'static, 'static>(&(), &());
// However, it is a hard error when not using method call syntax.
Foo::trait_method::<'static, 'static, 'static>(Foo, &(), &());
Foo::trait_function::<'static, 'static, 'static>(&(), &());
Foo::inherent_function::<'static, 'static, 'static>(&(), &());
free_function::<'static, 'static, 'static>(&(), &());
```

Even when specifying enough lifetime arguments for both the late and early bound lifetime parameter, these arguments are not actually used to annotate the lifetime provided to late bound parameters. We can demonstrate this by turbofishing `'static` to a function while providing a non-static borrow:
```rust
struct Foo;

impl Foo {
    fn inherent_method<'a: 'a, 'b>(self, _: &'a (), _: &'b String ) {}
}

Foo.inherent_method::<'static, 'static>(&(), &String::new());
```

This compiles even though the `&String::new()` function argument does not have a `'static` lifetime, this is because "extra" lifetime arguments are discarded rather than taken into account for late bound parameters when actually calling the function.

### Liveness of types with late bound parameters

When checking type outlives bounds involving function item types we take into account early bound parameters. For example:

```rust
fn foo<T>(_: T) {}

fn requires_static<T: 'static>(_: T) {}

fn bar<T>() {
    let f /* : FooFnItem<T> */ = foo::<T>;
    requires_static(f);
}
```

As the type parameter `T` is early bound, the desugaring of the function item type for `foo` would look something like `struct FooFnItem<T>`. Then in order for `FooFnItem<T>: 'static` to hold we must also require `T: 'static` to hold as otherwise we would wind up with soundness bugs.

Unfortunately, due to bugs in the compiler, we do not take into account early bound *lifetimes*, which is the cause of the open soundness bug [#84366](https://github.com/rust-lang/rust/issues/84366). This means that it's impossible to demonstrate a "difference" between early/late bound parameters for liveness/type outlives bounds as the only kind of generic parameters that are able to be late bound are lifetimes which are handled incorrectly.

Regardless, in theory the code example below *should* demonstrate such a difference once [#84366](https://github.com/rust-lang/rust/issues/84366) is fixed:
```rust
fn early_bound<'a: 'a>(_: &'a String) {}
fn late_bound<'a>(_: &'a String) {}

fn requires_static<T: 'static>(_: T) {}

fn bar<'b>() {
    let e = early_bound::<'b>;
    // this *should* error but does not
    requires_static(e);

    let l = late_bound;
    // this correctly does not error
    requires_static(l);
}
```

## Requirements for a parameter to be late bound

### Must be a lifetime parameter

Type and Const parameters are not able to be late bound as we do not have a way to support types such as `dyn for<T> Fn(Box<T>)` or `for<T> fn(Box<T>)`. Calling such types requires being able to monomorphize the underlying function which is not possible with indirection through dynamic dispatch.

### Must not be used in a where clause

Currently when a generic parameter is used in a where clause it must be early bound. For example:
```rust
# trait Trait<'a> {}
fn foo<'a, T: Trait<'a>>(_: &'a String, _: T) {}
```

In this example the lifetime parameter `'a` is considered to be early bound as it appears in the where clause `T: Trait<'a>`. This is true even for "trivial" where clauses such as `'a: 'a` or those implied by wellformedness of function arguments, for example:
```rust
fn foo<'a: 'a>(_: &'a String) {}
fn bar<'a, T: 'a>(_: &'a T) {}
```

In both of these functions the lifetime parameter `'a` would be considered to be early bound even though the where clauses they are used in arguably do not actually impose any constraints on the caller.

The reason for this restriction is a combination of two things:
- We cannot prove bounds on late bound parameters until they have been instantiated
- Function pointers and trait objects do not have a way to represent yet to be proven where clauses from the underlying function

Take the following example:
```rust
trait Trait<'a> {}
fn foo<'a, T: Trait<'a>>(_: &'a T) {}

let f = foo::<String>;
let f = f as for<'a> fn(&'a String);
f(&String::new());
```

At *some point* during type checking an error should be emitted for this code as `String` does not implement `Trait` for any lifetime.

If the lifetime `'a` were late bound then this becomes difficult to check. When naming `foo` we do not know what lifetime should be used as part of the `T: Trait<'a>` trait bound as it has not yet been instantiated. When coercing the function item type to a function pointer we have no way of tracking the `String: Trait<'a>` trait bound that must be proven when calling the function. 

If the lifetime `'a` is early bound (which it is in the current implementation in rustc), then the trait bound can be checked when naming the function `foo`. Requiring parameters used in where clauses to be early bound gives a natural place to check where clauses defined on the function.

Finally, we do not require lifetimes to be early bound if they are used in *implied bounds*, for example:
```rust
fn foo<'a, T>(_: &'a T) {}

let f = foo;
f(&String::new());
f(&String::new());
```

This code compiles, demonstrating that the lifetime parameter is late bound, even though `'a` is used in the type `&'a T` which implicitly requires `T: 'a` to hold. Implied bounds can be treated specially as any types introducing implied bounds are in the signature of the function pointer type, which means that when calling the function we know to prove `T: 'a`.

### Must be constrained by argument types

It is important that builtin impls on function item types do not wind up with unconstrained generic parameters as this can lead to unsoundness. This is the same kind of restriction as applies to user written impls, for example the following code results in an error:
```rust
trait Trait {
    type Assoc;
}

impl<'a> Trait for u8 {
    type Assoc = &'a String;
}
```

The analogous example for builtin impls on function items would be the following:
```rust,ignore
fn foo<'a>() -> &'a String { /* ... */ }
```
If the lifetime parameter `'a` were to be late bound we would wind up with a builtin impl with an unconstrained lifetime, we can manually write out the desugaring for the function item type and its impls with `'a` being late bound to demonstrate this:
```rust,ignore
// NOTE: this is just for demonstration, in practice `'a` is early bound
struct FooFnItem;

impl<'a> Fn<()> for FooFnItem {
    type Output = &'a String;
    /* fn call(...) -> ... { ... } */
}
```

In order to avoid such a situation we consider `'a` to be early bound which causes the lifetime on the impl to be constrained by the self type:
```rust,ignore
struct FooFnItem<'a>(PhantomData<fn() -> &'a String>);

impl<'a> Fn<()> for FooFnItem<'a> {
    type Output = &'a String;
    /* fn call(...) -> ... { ... } */
}
```
