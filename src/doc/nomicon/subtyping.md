% Subtyping and Variance

Although Rust doesn't have any notion of structural inheritance, it *does*
include subtyping. In Rust, subtyping derives entirely from lifetimes. Since
lifetimes are scopes, we can partially order them based on the *contains*
(outlives) relationship. We can even express this as a generic bound.

Subtyping on lifetimes is in terms of that relationship: if `'a: 'b` ("a contains
b" or "a outlives b"), then `'a` is a subtype of `'b`. This is a large source of
confusion, because it seems intuitively backwards to many: the bigger scope is a
*subtype* of the smaller scope.

This does in fact make sense, though. The intuitive reason for this is that if
you expect an `&'a u8`, then it's totally fine for me to hand you an `&'static
u8`, in the same way that if you expect an Animal in Java, it's totally fine for
me to hand you a Cat. Cats are just Animals *and more*, just as `'static` is
just `'a` *and more*.

(Note, the subtyping relationship and typed-ness of lifetimes is a fairly
arbitrary construct that some disagree with. However it simplifies our analysis
to treat lifetimes and types uniformly.)

Higher-ranked lifetimes are also subtypes of every concrete lifetime. This is
because taking an arbitrary lifetime is strictly more general than taking a
specific one.



# Variance

Variance is where things get a bit complicated.

Variance is a property that *type constructors* have with respect to their
arguments. A type constructor in Rust is a generic type with unbound arguments.
For instance `Vec` is a type constructor that takes a `T` and returns a
`Vec<T>`. `&` and `&mut` are type constructors that take two inputs: a
lifetime, and a type to point to.

A type constructor's *variance* is how the subtyping of its inputs affects the
subtyping of its outputs. There are two kinds of variance in Rust:

* F is *variant* over `T` if `T` being a subtype of `U` implies
  `F<T>` is a subtype of `F<U>` (subtyping "passes through")
* F is *invariant* over `T` otherwise (no subtyping relation can be derived)

(For those of you who are familiar with variance from other languages, what we
refer to as "just" variance is in fact *covariance*. Rust does not have
contravariance. Historically Rust did have some contravariance but it was
scrapped due to poor interactions with other features. If you experience
contravariance in Rust call your local compiler developer for medical advice.)

Some important variances:

* `&'a T` is variant over `'a` and `T` (as is `*const T` by metaphor)
* `&'a mut T` is variant with over `'a` but invariant over `T`
* `Fn(T) -> U` is invariant over `T`, but variant over `U`
* `Box`, `Vec`, and all other collections are variant over the types of
  their contents
* `UnsafeCell<T>`, `Cell<T>`, `RefCell<T>`, `Mutex<T>` and all other
  interior mutability types are invariant over T (as is `*mut T` by metaphor)

To understand why these variances are correct and desirable, we will consider
several examples.


We have already covered why `&'a T` should be variant over `'a` when
introducing subtyping: it's desirable to be able to pass longer-lived things
where shorter-lived things are needed.

Similar reasoning applies to why it should be variant over T. It is reasonable
to be able to pass `&&'static str` where an `&&'a str` is expected. The
additional level of indirection does not change the desire to be able to pass
longer lived things where shorted lived things are expected.

However this logic doesn't apply to `&mut`. To see why `&mut` should
be invariant over T, consider the following code:

```rust,ignore
fn overwrite<T: Copy>(input: &mut T, new: &mut T) {
    *input = *new;
}

fn main() {
    let mut forever_str: &'static str = "hello";
    {
        let string = String::from("world");
        overwrite(&mut forever_str, &mut &*string);
    }
    // Oops, printing free'd memory.
    println!("{}", forever_str);
}
```

The signature of `overwrite` is clearly valid: it takes mutable references to
two values of the same type, and overwrites one with the other. If `&mut T` was
variant over T, then `&mut &'static str` would be a subtype of `&mut &'a str`,
since `&'static str` is a subtype of `&'a str`. Therefore the lifetime of
`forever_str` would successfully be "shrunk" down to the shorter lifetime of
`string`, and `overwrite` would be called successfully. `string` would
subsequently be dropped, and `forever_str` would point to freed memory when we
print it! Therefore `&mut` should be invariant.

This is the general theme of variance vs invariance: if variance would allow you
to store a short-lived value into a longer-lived slot, then you must be
invariant.

However it *is* sound for `&'a mut T` to be variant over `'a`. The key difference
between `'a` and T is that `'a` is a property of the reference itself,
while T is something the reference is borrowing. If you change T's type, then
the source still remembers the original type. However if you change the
lifetime's type, no one but the reference knows this information, so it's fine.
Put another way: `&'a mut T` owns `'a`, but only *borrows* T.

`Box` and `Vec` are interesting cases because they're variant, but you can
definitely store values in them! This is where Rust gets really clever: it's
fine for them to be variant because you can only store values
in them *via a mutable reference*! The mutable reference makes the whole type
invariant, and therefore prevents you from smuggling a short-lived type into
them.

Being variant allows `Box` and `Vec` to be weakened when shared
immutably. So you can pass a `&Box<&'static str>` where a `&Box<&'a str>` is
expected.

However what should happen when passing *by-value* is less obvious. It turns out
that, yes, you can use subtyping when passing by-value. That is, this works:

```rust
fn get_box<'a>(str: &'a str) -> Box<&'a str> {
    // string literals are `&'static str`s.
    Box::new("hello")
}
```

Weakening when you pass by-value is fine because there's no one else who
"remembers" the old lifetime in the Box. The reason a variant `&mut` was
trouble was because there's always someone else who remembers the original
subtype: the actual owner.

The invariance of the cell types can be seen as follows: `&` is like an `&mut`
for a cell, because you can still store values in them through an `&`. Therefore
cells must be invariant to avoid lifetime smuggling.

`Fn` is the most subtle case because it has mixed variance. To see why
`Fn(T) -> U` should be invariant over T, consider the following function
signature:

```rust,ignore
// 'a is derived from some parent scope.
fn foo(&'a str) -> usize;
```

This signature claims that it can handle any `&str` that lives at least as
long as `'a`. Now if this signature was variant over `&'a str`, that
would mean

```rust,ignore
fn foo(&'static str) -> usize;
```

could be provided in its place, as it would be a subtype. However this function
has a stronger requirement: it says that it can only handle `&'static str`s,
and nothing else. Giving `&'a str`s to it would be unsound, as it's free to
assume that what it's given lives forever. Therefore functions are not variant
over their arguments.

To see why `Fn(T) -> U` should be variant over U, consider the following
function signature:

```rust,ignore
// 'a is derived from some parent scope.
fn foo(usize) -> &'a str;
```

This signature claims that it will return something that outlives `'a`. It is
therefore completely reasonable to provide

```rust,ignore
fn foo(usize) -> &'static str;
```

in its place. Therefore functions are variant over their return type.

`*const` has the exact same semantics as `&`, so variance follows. `*mut` on the
other hand can dereference to an `&mut` whether shared or not, so it is marked
as invariant just like cells.

This is all well and good for the types the standard library provides, but
how is variance determined for type that *you* define? A struct, informally
speaking, inherits the variance of its fields. If a struct `Foo`
has a generic argument `A` that is used in a field `a`, then Foo's variance
over `A` is exactly `a`'s variance. However this is complicated if `A` is used
in multiple fields.

* If all uses of A are variant, then Foo is variant over A
* Otherwise, Foo is invariant over A

```rust
use std::cell::Cell;

struct Foo<'a, 'b, A: 'a, B: 'b, C, D, E, F, G, H> {
    a: &'a A,     // Variant over 'a and A.
    b: &'b mut B, // Invariant over 'b and B.
    c: *const C,  // Variant over C.
    d: *mut D,    // Invariant over D.
    e: Vec<E>,    // Variant over E.
    f: Cell<F>,   // Invariant over F.
    g: G,         // Variant over G.
    h1: H,        // Would also be variant over H except...
    h2: Cell<H>,  // Invariant over H, because invariance wins.
}
```
