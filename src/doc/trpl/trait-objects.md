% Trait Objects

When code involves polymorphism, there needs to be a mechanism to determine
which specific version is actually run. This is called ‘dispatch’. There are
two major forms of dispatch: static dispatch and dynamic dispatch. While Rust
favors static dispatch, it also supports dynamic dispatch through a mechanism
called ‘trait objects’.

## Background

For the rest of this chapter, we’ll need a trait and some implementations.
Let’s make a simple one, `Foo`. It has one method that is expected to return a
`String`.

```rust
trait Foo {
    fn method(&self) -> String;
}
```

We’ll also implement this trait for `u8` and `String`:

```rust
# trait Foo { fn method(&self) -> String; }
impl Foo for u8 {
    fn method(&self) -> String { format!("u8: {}", *self) }
}

impl Foo for String {
    fn method(&self) -> String { format!("string: {}", *self) }
}
```


## Static dispatch

We can use this trait to perform static dispatch with trait bounds:

```rust
# trait Foo { fn method(&self) -> String; }
# impl Foo for u8 { fn method(&self) -> String { format!("u8: {}", *self) } }
# impl Foo for String { fn method(&self) -> String { format!("string: {}", *self) } }
fn do_something<T: Foo>(x: T) {
    x.method();
}

fn main() {
    let x = 5u8;
    let y = "Hello".to_string();

    do_something(x);
    do_something(y);
}
```

Rust uses ‘monomorphization’ to perform static dispatch here. This means that
Rust will create a special version of `do_something()` for both `u8` and
`String`, and then replace the call sites with calls to these specialized
functions. In other words, Rust generates something like this:

```rust
# trait Foo { fn method(&self) -> String; }
# impl Foo for u8 { fn method(&self) -> String { format!("u8: {}", *self) } }
# impl Foo for String { fn method(&self) -> String { format!("string: {}", *self) } }
fn do_something_u8(x: u8) {
    x.method();
}

fn do_something_string(x: String) {
    x.method();
}

fn main() {
    let x = 5u8;
    let y = "Hello".to_string();

    do_something_u8(x);
    do_something_string(y);
}
```

This has a great upside: static dispatch allows function calls to be
inlined because the callee is known at compile time, and inlining is
the key to good optimization. Static dispatch is fast, but it comes at
a tradeoff: ‘code bloat’, due to many copies of the same function
existing in the binary, one for each type.

Furthermore, compilers aren’t perfect and may “optimize” code to become slower.
For example, functions inlined too eagerly will bloat the instruction cache
(cache rules everything around us). This is part of the reason that `#[inline]`
and `#[inline(always)]` should be used carefully, and one reason why using a
dynamic dispatch is sometimes more efficient.

However, the common case is that it is more efficient to use static dispatch,
and one can always have a thin statically-dispatched wrapper function that does
a dynamic dispatch, but not vice versa, meaning static calls are more flexible.
The standard library tries to be statically dispatched where possible for this
reason.

## Dynamic dispatch

Rust provides dynamic dispatch through a feature called ‘trait objects’. Trait
objects, like `&Foo` or `Box<Foo>`, are normal values that store a value of
*any* type that implements the given trait, where the precise type can only be
known at runtime.

A trait object can be obtained from a pointer to a concrete type that
implements the trait by *casting* it (e.g. `&x as &Foo`) or *coercing* it
(e.g. using `&x` as an argument to a function that takes `&Foo`).

These trait object coercions and casts also work for pointers like `&mut T` to
`&mut Foo` and `Box<T>` to `Box<Foo>`, but that’s all at the moment. Coercions
and casts are identical.

This operation can be seen as ‘erasing’ the compiler’s knowledge about the
specific type of the pointer, and hence trait objects are sometimes referred to
as ‘type erasure’.

Coming back to the example above, we can use the same trait to perform dynamic
dispatch with trait objects by casting:

```rust
# trait Foo { fn method(&self) -> String; }
# impl Foo for u8 { fn method(&self) -> String { format!("u8: {}", *self) } }
# impl Foo for String { fn method(&self) -> String { format!("string: {}", *self) } }

fn do_something(x: &Foo) {
    x.method();
}

fn main() {
    let x = 5u8;
    do_something(&x as &Foo);
}
```

or by coercing:

```rust
# trait Foo { fn method(&self) -> String; }
# impl Foo for u8 { fn method(&self) -> String { format!("u8: {}", *self) } }
# impl Foo for String { fn method(&self) -> String { format!("string: {}", *self) } }

fn do_something(x: &Foo) {
    x.method();
}

fn main() {
    let x = "Hello".to_string();
    do_something(&x);
}
```

A function that takes a trait object is not specialized to each of the types
that implements `Foo`: only one copy is generated, often (but not always)
resulting in less code bloat. However, this comes at the cost of requiring
slower virtual function calls, and effectively inhibiting any chance of
inlining and related optimizations from occurring.

### Why pointers?

Rust does not put things behind a pointer by default, unlike many managed
languages, so types can have different sizes. Knowing the size of the value at
compile time is important for things like passing it as an argument to a
function, moving it about on the stack and allocating (and deallocating) space
on the heap to store it.

For `Foo`, we would need to have a value that could be at least either a
`String` (24 bytes) or a `u8` (1 byte), as well as any other type for which
dependent crates may implement `Foo` (any number of bytes at all). There’s no
way to guarantee that this last point can work if the values are stored without
a pointer, because those other types can be arbitrarily large.

Putting the value behind a pointer means the size of the value is not relevant
when we are tossing a trait object around, only the size of the pointer itself.

### Representation

The methods of the trait can be called on a trait object via a special record
of function pointers traditionally called a ‘vtable’ (created and managed by
the compiler).

Trait objects are both simple and complicated: their core representation and
layout is quite straight-forward, but there are some curly error messages and
surprising behaviors to discover.

Let’s start simple, with the runtime representation of a trait object. The
`std::raw` module contains structs with layouts that are the same as the
complicated built-in types, [including trait objects][stdraw]:

```rust
# mod foo {
pub struct TraitObject {
    pub data: *mut (),
    pub vtable: *mut (),
}
# }
```

[stdraw]: ../std/raw/struct.TraitObject.html

That is, a trait object like `&Foo` consists of a ‘data’ pointer and a ‘vtable’
pointer.

The data pointer addresses the data (of some unknown type `T`) that the trait
object is storing, and the vtable pointer points to the vtable (‘virtual method
table’) corresponding to the implementation of `Foo` for `T`.


A vtable is essentially a struct of function pointers, pointing to the concrete
piece of machine code for each method in the implementation. A method call like
`trait_object.method()` will retrieve the correct pointer out of the vtable and
then do a dynamic call of it. For example:

```rust,ignore
struct FooVtable {
    destructor: fn(*mut ()),
    size: usize,
    align: usize,
    method: fn(*const ()) -> String,
}

// u8:

fn call_method_on_u8(x: *const ()) -> String {
    // the compiler guarantees that this function is only called
    // with `x` pointing to a u8
    let byte: &u8 = unsafe { &*(x as *const u8) };

    byte.method()
}

static Foo_for_u8_vtable: FooVtable = FooVtable {
    destructor: /* compiler magic */,
    size: 1,
    align: 1,

    // cast to a function pointer
    method: call_method_on_u8 as fn(*const ()) -> String,
};


// String:

fn call_method_on_String(x: *const ()) -> String {
    // the compiler guarantees that this function is only called
    // with `x` pointing to a String
    let string: &String = unsafe { &*(x as *const String) };

    string.method()
}

static Foo_for_String_vtable: FooVtable = FooVtable {
    destructor: /* compiler magic */,
    // values for a 64-bit computer, halve them for 32-bit ones
    size: 24,
    align: 8,

    method: call_method_on_String as fn(*const ()) -> String,
};
```

The `destructor` field in each vtable points to a function that will clean up
any resources of the vtable’s type: for `u8` it is trivial, but for `String` it
will free the memory. This is necessary for owning trait objects like
`Box<Foo>`, which need to clean-up both the `Box` allocation as well as the
internal type when they go out of scope. The `size` and `align` fields store
the size of the erased type, and its alignment requirements; these are
essentially unused at the moment since the information is embedded in the
destructor, but will be used in the future, as trait objects are progressively
made more flexible.

Suppose we’ve got some values that implement `Foo`. The explicit form of
construction and use of `Foo` trait objects might look a bit like (ignoring the
type mismatches: they’re all just pointers anyway):

```rust,ignore
let a: String = "foo".to_string();
let x: u8 = 1;

// let b: &Foo = &a;
let b = TraitObject {
    // store the data
    data: &a,
    // store the methods
    vtable: &Foo_for_String_vtable
};

// let y: &Foo = x;
let y = TraitObject {
    // store the data
    data: &x,
    // store the methods
    vtable: &Foo_for_u8_vtable
};

// b.method();
(b.vtable.method)(b.data);

// y.method();
(y.vtable.method)(y.data);
```

## Object Safety

Not every trait can be used to make a trait object. For example, vectors implement
`Clone`, but if we try to make a trait object:

```ignore
let v = vec![1, 2, 3];
let o = &v as &Clone;
```

We get an error:

```text
error: cannot convert to a trait object because trait `core::clone::Clone` is not object-safe [E0038]
let o = &v as &Clone;
        ^~
note: the trait cannot require that `Self : Sized`
let o = &v as &Clone;
        ^~
```

The error says that `Clone` is not ‘object-safe’. Only traits that are
object-safe can be made into trait objects. A trait is object-safe if both of
these are true:

* the trait does not require that `Self: Sized`
* all of its methods are object-safe

So what makes a method object-safe? Each method must require that `Self: Sized`
or all of the following:

* must not have any type parameters
* must not use `Self`

Whew! As we can see, almost all of these rules talk about `Self`. A good intuition
is “except in special circumstances, if your trait’s method uses `Self`, it is not
object-safe.”
