% Subtyping and Variance

Although Rust doesn't have any notion of inheritance, it *does* include subtyping.
In Rust, subtyping derives entirely from *lifetimes*. Since lifetimes are scopes,
we can partially order them based on a *contains* (outlives) relationship. We
can even express this as a generic bound: `T: 'a` specifies that whatever scope `T`
is valid for must contain the scope `'a` ("T outlives `'a`").

We can then define subtyping on lifetimes in terms of that relationship: if `'a: 'b`
("a contains b" or "a outlives b"), then `'a` is a subtype of `'b`. This is a
large source of confusion, because it seems intuitively backwards to many:
the bigger scope is a *sub type* of the smaller scope.

This does in fact make sense. The intuitive reason for this is that if you expect an
`&'a u8`, then it's totally fine for me to hand you an `&'static u8`, in the same way
that if you expect an Animal in Java, it's totally fine for me to hand you a Cat.
Cats are just Animals *and more*, just as `'static` is just `'a` *and more*.

(Note, the subtyping relationship and typed-ness of lifetimes is a fairly arbitrary
construct that some disagree with. I just find that it simplifies this analysis.)

Higher-ranked lifetimes are also subtypes of every concrete lifetime. This is because
taking an arbitrary lifetime is strictly more general than taking a specific one.



# Variance

Variance is where things get really harsh.

Variance is a property that *type constructors* have. A type constructor in Rust
is a generic type with unbound arguments. For instance `Vec` is a type constructor
that takes a `T` and returns a `Vec<T>`. `&` and `&mut` are type constructors that
take a lifetime and a type.

A type constructor's *variance* is how the subtypes of its inputs affects the
subtypes of its outputs. There are three kinds of variance:

* F is *variant* if `T` being a subtype of `U` implies `F<T>` is a subtype of `F<U>`
* F is *invariant* otherwise (no subtyping relation can be derived)

(For those of you who are familiar with variance from other languages, what we refer
to as "just" variance is in fact *covariance*. Rust does not have contravariance.
Historically Rust did have some contravariance but it was scrapped due to poor
interactions with other features.)

Some important variances:

* `&` is variant (as is `*const` by metaphor)
* `&mut` is invariant (as is `*mut` by metaphor)
* `Fn(T) -> U` is invariant with respect to `T`, but variant with respect to `U`
* `Box`, `Vec`, and all other collections are variant
* `UnsafeCell`, `Cell`, `RefCell`, `Mutex` and all "interior mutability"
  types are invariant

To understand why these variances are correct and desirable, we will consider several
examples. We have already covered why `&` should be variant when introducing subtyping:
it's desirable to be able to pass longer-lived things where shorter-lived things are
needed.

To see why `&mut` should be invariant, consider the following code:

```rust
fn main() {
    let mut forever_str: &'static str = "hello";
    {
        let string = String::from("world");
        overwrite(&mut forever_str, &mut &*string);
    }
    println!("{}", forever_str);
}

fn overwrite<T: Copy>(input: &mut T, new: &mut T) {
    *input = *new;
}
```

The signature of `overwrite` is clearly valid: it takes mutable references to two values
of the same type, and overwrites one with the other. We have seen already that `&` is
variant, and `'static` is a subtype of *any* `'a`, so `&'static str` is a
subtype of `&'a str`. Therefore, if `&mut` was
*also* variant, then the lifetime of the `&'static str` would successfully be
"shrunk" down to the shorter lifetime of the string, and `overwrite` would be
called successfully. The string would subsequently be dropped, and `forever_str`
would point to freed memory when we print it!

Therefore `&mut` should be invariant. This is the general theme of variance vs
invariance: if variance would allow you to *store* a short-lived value in a
longer-lived slot, then you must be invariant.

`Box` and `Vec` are interesting cases because they're variant, but you can
definitely store values in them! This is fine because *you can only store values
in them through a mutable reference*! The mutable reference makes the whole type
invariant, and therefore prevents you from getting in trouble.

Being variant allows them to be variant when shared immutably (so you can pass
a `&Box<&'static str>` where a `&Box<&'a str>` is expected). It also allows you to
forever weaken the type by moving it into a weaker slot. That is, you can do:

```rust
fn get_box<'a>(&'a u8) -> Box<&'a str> {
    // string literals are `&'static str`s
    Box::new("hello")
}
```

which is fine because unlike the mutable borrow case, there's no one else who
"remembers" the old lifetime in the box.

The variance of the cell types similarly follows. `&` is like an `&mut` for a
cell, because you can still store values in them through an `&`. Therefore cells
must be invariant to avoid lifetime smuggling.

`Fn` is the most subtle case, because it has mixed variance. To see why
`Fn(T) -> U` should be invariant over T, consider the following function
signature:

```rust
// 'a is derived from some parent scope
fn foo(&'a str) -> usize;
```

This signature claims that it can handle any &str that lives *at least* as long
as `'a`. Now if this signature was variant with respect to `&str`, that would mean

```rust
fn foo(&'static str) -> usize;
```

could be provided in its place, as it would be a subtype. However this function
has a *stronger* requirement: it says that it can *only* handle `&'static str`s,
and nothing else. Therefore functions are not variant over their arguments.

To see why `Fn(T) -> U` should be *variant* over U, consider the following
function signature:

```rust
// 'a is derived from some parent scope
fn foo(usize) -> &'a str;
```

This signature claims that it will return something that outlives `'a`. It is
therefore completely reasonable to provide

```rust
fn foo(usize) -> &'static str;
```

in its place. Therefore functions *are* variant over their return type.

`*const` has the exact same semantics as `&`, so variance follows. `*mut` on the
other hand can dereference to an &mut whether shared or not, so it is marked
as invariant in analogy to cells.

This is all well and good for the types the standard library provides, but
how is variance determined for type that *you* define? A struct, informally
speaking, inherits the variance of its fields. If a struct `Foo`
has a generic argument `A` that is used in a field `a`, then Foo's variance
over `A` is exactly `a`'s variance. However this is complicated if `A` is used
in multiple fields.

* If all uses of A are variant, then Foo is variant over A
* Otherwise, Foo is invariant over A

```rust
struct Foo<'a, 'b, A, B, C, D, E, F, G, H> {
    a: &'a A,     // variant over 'a and A
    b: &'b mut B, // invariant over 'b and B
    c: *const C,  // variant over C
    d: *mut D,    // invariant over D
    e: Vec<E>,    // variant over E
    f: Cell<F>,   // invariant over F
    g: G          // variant over G
    h1: H         // would also be variant over H except...
    h2: Cell<H>   // invariant over H, because invariance wins
}
```