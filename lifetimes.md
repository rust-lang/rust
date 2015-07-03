% Ownership and Lifetimes

Ownership is the breakout feature of Rust. It allows Rust to be completely
memory-safe and efficient, while avoiding garbage collection. Before getting
into the ownership system in detail, we will consider the motivation of this
design.

TODO: Interior Mutability section




# Living Without Garbage Collection

We will assume that you accept that garbage collection is not always an optimal
solution, and that it is desirable to manually manage memory to some extent.
If you do not accept this, might I interest you in a different language?

Regardless of your feelings on GC, it is pretty clearly a *massive* boon to
making code safe. You never have to worry about things going away *too soon*
(although whether you still *wanted* to be pointing at that thing is a different
issue...). This is a pervasive problem that C and C++ need to deal with.
Consider this simple mistake that all of us who have used a non-GC'd language
have made at one point:

```rust,ignore
fn as_str(data: &u32) -> &str {
    // compute the string
    let s = format!("{}", data);

    // OH NO! We returned a reference to something that
    // exists only in this function!
    // Dangling pointer! Use after free! Alas!
    // (this does not compile in Rust)
    &s
}
```

This is exactly what Rust's ownership system was built to solve.
Rust knows the scope in which the `&s` lives, and as such can prevent it from
escaping. However this is a simple case that even a C compiler could plausibly
catch. Things get more complicated as code gets bigger and pointers get fed through
various functions. Eventually, a C compiler will fall down and won't be able to
perform sufficient escape analysis to prove your code unsound. It will consequently
be forced to accept your program on the assumption that it is correct.

This will never happen to Rust. It's up to the programmer to prove to the
compiler that everything is sound.

Of course, rust's story around ownership is much more complicated than just
verifying that references don't escape the scope of their referrent. That's
because ensuring pointers are always valid is much more complicated than this.
For instance in this code,

```rust,ignore
let mut data = vec![1, 2, 3];
// get an internal reference
let x = &data[0];

// OH NO! `push` causes the backing storage of `data` to be reallocated.
// Dangling pointer! User after free! Alas!
// (this does not compile in Rust)
data.push(4);

println!("{}", x);
```

naive scope analysis would be insufficient to prevent this bug, because `data`
does in fact live as long as we needed. However it was *changed* while we had
a reference into it. This is why Rust requires any references to freeze the
referrent and its owners.



# References

There are two kinds of reference:

* Shared reference: `&`
* Mutable reference: `&mut`

Which obey the following rules:

* A reference cannot outlive its referrent
* A mutable reference cannot be aliased

To define aliasing, we must define the notion of *paths* and *liveness*.




## Paths

If all Rust had were values, then every value would be uniquely owned
by a variable or composite structure. From this we naturally derive a *tree*
of ownership. The stack itself is the root of the tree, with every variable
as its direct children. Each variable's direct children would be their fields
(if any), and so on.

From this view, every value in Rust has a unique *path* in the tree of ownership.
References to a value can subsequently be interpretted as a path in this tree.
Of particular interest are *prefixes*: `x` is a prefix of `y` if `x` owns `y`

However much data doesn't reside on the stack, and we must also accomodate this.
Globals and thread-locals are simple enough to model as residing at the bottom
of the stack. However data on the heap poses a different problem.

If all Rust had on the heap was data uniquely by a pointer on the stack,
then we can just treat that pointer as a struct that owns the value on
the heap. Box, Vec, String, and HashMap, are examples of types which uniquely
own data on the heap.

Unfortunately, data on the heap is not *always* uniquely owned. Rc for instance
introduces a notion of *shared* ownership. Shared ownership means there is no
unique path. A value with no unique path limits what we can do with it. In general, only
shared references can be created to these values. However mechanisms which ensure
mutual exclusion may establish One True Owner temporarily, establishing a unique path
to that value (and therefore all its children).

The most common way to establish such a path is through *interior mutability*,
in contrast to the *inherited mutability* that everything in Rust normally uses.
Cell, RefCell, Mutex, and RWLock are all examples of interior mutability types. These
types provide exclusive access through runtime restrictions. However it is also
possible to establish unique ownership without interior mutability. For instance,
if an Rc has refcount 1, then it is safe to mutate or move its internals.




## Liveness

Roughly, a reference is *live* at some point in a program if it can be
dereferenced. Shared references are always live unless they are literally unreachable
(for instance, they reside in freed or leaked memory). Mutable references can be
reachable but *not* live through the process of *reborrowing*.

A mutable reference can be reborrowed to either a shared or mutable reference.
Further, the reborrow can produce exactly the same reference, or point to a
path it is a prefix of. For instance, a mutable reference can be reborrowed
to point to a field of its referrent:

```rust
let x = &mut (1, 2);
{
    // reborrow x to a subfield
    let y = &mut x.0;
    // y is now live, but x isn't
    *y = 3;
}
// y goes out of scope, so x is live again
*x = (5, 7);
```

It is also possible to reborrow into *multiple* mutable references, as long as
they are to *disjoint*: no reference is a prefix of another. Rust
explicitly enables this to be done with disjoint struct fields, because
disjointness can be statically proven:

```
let x = &mut (1, 2);
{
    // reborrow x to two disjoint subfields
    let y = &mut x.0;
    let z = &mut x.1;
    // y and z are now live, but x isn't
    *y = 3;
    *z = 4;
}
// y and z go out of scope, so x is live again
*x = (5, 7);
```

However it's often the case that Rust isn't sufficiently smart to prove that
multiple borrows are disjoint. *This does not mean it is fundamentally illegal
to make such a borrow*, just that Rust isn't as smart as you want.

To simplify things, we can model variables as a fake type of reference: *owned*
references. Owned references have much the same semantics as mutable references:
they can be re-borrowed in a mutable or shared manner, which makes them no longer
live. Live owned references have the unique property that they can be moved
out of (though mutable references *can* be swapped out of). This is
only given to *live* owned references because moving its referrent would of
course invalidate all outstanding references prematurely.

As a local lint against inappropriate mutation, only variables that are marked
as `mut` can be borrowed mutably.

It is also interesting to note that Box behaves exactly like an owned
reference. It can be moved out of, and Rust understands it sufficiently to
reason about its paths like a normal variable.




## Aliasing

With liveness and paths defined, we can now properly define *aliasing*:

**A mutable reference is aliased if there exists another live reference to it or
one of its prefixes.**

That's it. Super simple right? Except for the fact that it took us two pages
to define all of the terms in that defintion. You know: Super. Simple.

Actually it's a bit more complicated than that. In addition to references,
Rust has *raw pointers*: `*const T` and `*mut T`. Raw pointers have no inherent
ownership or aliasing semantics. As a result, Rust makes absolutely no effort
to track that they are used correctly, and they are wildly unsafe.

**It is an open question to what degree raw pointers have alias semantics.
However it is important for these definitions to be sound that the existence
of a raw pointer does not imply some kind of live path.**




# Lifetimes

Rust enforces these rules through *lifetimes*. Lifetimes are effectively
just names for scopes on the stack, somewhere in the program. Each reference,
and anything that contains a reference, is tagged with a lifetime specifying
the scope it's valid for.

Within a function body, Rust generally doesn't let you explicitly name the
lifetimes involved. This is because it's generally not really *necessary*
to talk about lifetimes in a local context; rust has all the information and
can work out everything.

However once you cross the function boundary, you need to start talking about
lifetimes. Lifetimes are denoted with an apostrophe: `'a`, `'static`. To dip
our toes with lifetimes, we're going to pretend that we're actually allowed
to label scopes with lifetimes, and desugar the examples from the start of
this chapter.

Our examples made use of *aggressive* sugar around scopes and lifetimes,
because writing everything out explicitly is *extremely noisy*. All rust code
relies on aggressive inference and elision of "obvious" things.

One particularly interesting piece of sugar is that each `let` statement implicitly
introduces a scope. For the most part, this doesn't really matter. However it
does matter for variables that refer to each other. As a simple example, let's
completely desugar this simple piece of Rust code:

```rust
let x = 0;
let y = &x;
let z = &y;
```

becomes:

```rust,ignore
// NOTE: `'a:` and `&'a x` is not valid syntax!
'a: {
    let x: i32 = 0;
    'b: {
        let y: &'a i32 = &'a x;
        'c: {
            let z: &'b &'a i32 = &'b y;
        }
    }
}
```

Wow. That's... awful. Let's all take a moment to thank Rust for being a huge
pile of sugar with sugar on top.

Anyway, let's look at some of those examples from before:

```rust,ignore
fn as_str(data: &u32) -> &str {
    let s = format!("{}", data);
    &s
}
```

desugars to:

```rust,ignore
fn as_str<'a>(data: &'a u32) -> &'a str {
    'b: {
        let s = format!("{}", data);
        return &'b s
    }
}
```

This signature of `as_str` takes a reference to a u32 with *some* lifetime, and
promises that it can produce a reference to a str that can live *just as long*.
Already we can see why this signature might be trouble. That basically implies
that we're going to *find* a str somewhere in the scope that u32 originated in,
or somewhere *even* earlier. That's uh... a big ask.

We then proceed to compute the string `s`, and return a reference to it.
Unfortunately, since `s` was defined in the scope `'b`, the reference we're
returning can only live for that long. From the perspective of the compiler,
we've failed *twice* here. We've failed to fulfill the contract we were asked
to fulfill (`'b` is unrelated to `'a`); and we've also tried to make a reference
outlive its referrent by returning an `&'b`, where `'b` is in our function.

Shoot!

Of course, the right way to right this function is as follows:

```rust
fn to_string(data: &u32) -> String {
    format!("{}", data)
}
```

We must produce an owned value inside the function to return it! The only way
we could have returned an `&'a str` would have been if it was in a field of the
`&'a u32`, which is obviously not the case.

(Actually we could have also just returned a string literal, though this limits
the behaviour of our function *just a bit*.)

How about the other example:

```rust,ignore
let mut data = vec![1, 2, 3];
let x = &data[0];
data.push(4);
println!("{}", x);
```

```rust,ignore
'a: {
    let mut data: Vec<i32> = vec![1, 2, 3];
    'b: {
        let x: &'a i32 = Index::index(&'a data, 0);
        'c: {
            // Exactly what the desugar for Vec::push is is up to Rust.
            // This particular desugar is a decent approximation for our
            // purpose. In particular methods oft invoke a temporary borrow.
            let temp: &'c mut Vec = &'c mut data;
            // NOTE: Vec::push is not valid syntax
            Vec::push(temp, 4);
        }
        println!("{}", x);
    }
}
```

Here the problem is that we're trying to mutably borrow the `data` path, while
we have a reference into something it's a prefix of. Rust subsequently throws
up its hands in disgust and rejects our program. The correct way to write this
is to just re-order the code so that we make `x` *after* we push:

TODO: convince myself of this.

```rust
let mut data = vec![1, 2, 3];
data.push(4);

let x = &data[0];
println!("{}", x);
```



# Lifetime Elision

In order to make common patterns more ergonomic, Rust allows lifetimes to be
*elided* in function signatures.

A *lifetime position* is anywhere you can write a lifetime in a type:

```rust
&'a T
&'a mut T
T<'a>
```

Lifetime positions can appear as either "input" or "output":

* For `fn` definitions, input refers to the types of the formal arguments
  in the `fn` definition, while output refers to
  result types. So `fn foo(s: &str) -> (&str, &str)` has elided one lifetime in
  input position and two lifetimes in output position.
  Note that the input positions of a `fn` method definition do not
  include the lifetimes that occur in the method's `impl` header
  (nor lifetimes that occur in the trait header, for a default method).

* In the future, it should be possible to elide `impl` headers in the same manner.

Elision rules are as follows:

* Each elided lifetime in input position becomes a distinct lifetime
  parameter.

* If there is exactly one input lifetime position (elided or not), that lifetime
  is assigned to *all* elided output lifetimes.

* If there are multiple input lifetime positions, but one of them is `&self` or
  `&mut self`, the lifetime of `self` is assigned to *all* elided output lifetimes.

* Otherwise, it is an error to elide an output lifetime.

Examples:

```rust
fn print(s: &str);                                      // elided
fn print<'a>(s: &'a str);                               // expanded

fn debug(lvl: uint, s: &str);                           // elided
fn debug<'a>(lvl: uint, s: &'a str);                    // expanded

fn substr(s: &str, until: uint) -> &str;                // elided
fn substr<'a>(s: &'a str, until: uint) -> &'a str;      // expanded

fn get_str() -> &str;                                   // ILLEGAL

fn frob(s: &str, t: &str) -> &str;                      // ILLEGAL

fn get_mut(&mut self) -> &mut T;                        // elided
fn get_mut<'a>(&'a mut self) -> &'a mut T;              // expanded

fn args<T:ToCStr>(&mut self, args: &[T]) -> &mut Command                  // elided
fn args<'a, 'b, T:ToCStr>(&'a mut self, args: &'b [T]) -> &'a mut Command // expanded

fn new(buf: &mut [u8]) -> BufWriter;                    // elided
fn new<'a>(buf: &'a mut [u8]) -> BufWriter<'a>          // expanded

```



# Unbounded Lifetimes

Unsafe code can often end up producing references or lifetimes out of thin air.
Such lifetimes come into the world as *unbounded*. The most common source of this
is derefencing a raw pointer, which produces a reference with an unbounded lifetime.
Such a lifetime becomes as big as context demands. This is in fact more powerful
than simply becoming `'static`, because for instance `&'static &'a T`
will fail to typecheck, but the unbound lifetime will perfectly mold into
`&'a &'a T` as needed. However for most intents and purposes, such an unbounded
lifetime can be regarded as `'static`.

Almost no reference is `'static`, so this is probably wrong. `transmute` and
`transmute_copy` are the two other primary offenders. One should endeavour to
bound an unbounded lifetime as quick as possible, especially across function
boundaries.

Given a function, any output lifetimes that don't derive from inputs are
unbounded. For instance:

```
fn get_str<'a>() -> &'a str;
```

will produce an `&str` with an unbounded lifetime. The easiest way to avoid
unbounded lifetimes is to use lifetime elision at the function boundary.
If an output lifetime is elided, then it *must* be bounded by an input lifetime.
Of course, it might be bounded by the *wrong* lifetime, but this will usually
just cause a compiler error, rather than allow memory safety to be trivially
violated.

Within a function, bounding lifetimes is more error-prone. The safest and easiest
way to bound a lifetime is to return it from a function with a bound lifetime.
However if this is unacceptable, the reference can be placed in a location with
a specific lifetime. Unfortunately it's impossible to name all lifetimes involved
in a function. To get around this, you can in principle use `copy_lifetime`, though
these are unstable due to their awkward nature and questionable utility.





# Higher-Rank Trait Bounds

// TODO: make aturon less mad

Generics in Rust generally allow types to be instantiated with arbitrary
associated lifetimes, but this fixes the lifetimes they work with once
instantiated. For almost all types, this is exactly the desired behaviour.
For example slice::Iter can work with arbitrary lifetimes, determined by the
slice that instantiates it. However *once* Iter is instantiated the lifetimes
it works with cannot be changed. It returns references that live for some
particular `'a`.

However some types are more flexible than this. In particular, a single
instantiation of a function can process arbitrary lifetimes:

```rust
fn identity(input: &u8) -> &u8 { input }
```

What is *the* lifetime that identity works with? There is none. If you think
this is "cheating" because functions are statically instantiated, then you need
only consider the equivalent closure:

```rust
let identity = |input: &u8| input;
```

These functions are *higher ranked* over the lifetimes they work with. This means
that they're generic over what they handle *after instantiation*. For most things
this would pose a massive problem, but because lifetimes don't *exist* at runtime,
this is really just a compile-time mechanism. The Fn traits contain sugar that
allows higher-rank lifetimes to simply be expressed by simply omitting lifetimes:


```rust
fn main() {
    foo(|input| input);
}

fn foo<F>(f: F)
    // F is higher-ranked over the lifetime these references have
    where F: Fn(&u8) -> &u8
{
    f(&0);
    f(&1);
}
```

The desugaring of this is actually unstable:

```
#![feature(unboxed_closures)]

fn main() {
    foo(|input| input);
}

fn foo<F>(f: F)
    where F: for<'a> Fn<(&'a u8,), Output=&'a u8>
{
    f(&0);
    f(&1);
}
```

`for<'a>` is how we declare a higher-ranked lifetime. Unfortunately higher-ranked
lifetimes are still fairly new, and are missing a few features to make them
maximally useful outside of the Fn traits.




# Subtyping and Variance

Although Rust doesn't have any notion of inheritance, it *does* include subtyping.
In Rust, subtyping derives entirely from *lifetimes*. Since lifetimes are derived
from scopes, we can partially order them based on an *outlives* relationship. We
can even express this as a generic bound: `T: 'a` specifies that `T` *outlives* `'a`.

We can then define subtyping on lifetimes in terms of lifetimes: if `'a : 'b`
("a outlives b"), then `'a` is a subtype of  `b`. This is a
large source of confusion, because a bigger scope is a *sub type* of a smaller scope.
This does in fact make sense. The intuitive reason for this is that if you expect an
`&'a u8`, then it's totally fine for me to hand you an `&'static u8` in the same way
that if you expect an Animal in Java, it's totally fine for me to hand you a Cat.

(Note, the subtyping relationship and typed-ness of lifetimes is a fairly arbitrary
construct that some disagree with. I just find that it simplifies this analysis.)

TODO: higher rank lifetime subtyping

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
to as "just" variant is in fact *covariant*. Rust does not have contravariance.
Historically Rust did have some contravariance but it was scrapped due to poor
interactions with other features.)

Some important variances:

* `&` is variant (as is *const by metaphor)
* `&mut` is invariant (as is *mut by metaphor)
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
of the same type, and replaces one with the other. We have seen already that `&` is
variant, and `'static` is a subtype of *any* `'a`, so `&'static str` is a
subtype of `&'a str`. Therefore, if `&mut` was
*also* variant, then the lifetime of the `&'static str` would successfully be
"shrunk" down to the shorter lifetime of the string, and `replace` would be
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
as `'a`. Now if this signature was variant with respect to &str, that would mean

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

`*const` has the exact same semantics as &, so variance follows. `*mut` on the
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



# PhantomData

When working with unsafe code, we can often end up in a situation where
types or lifetimes are logically associated with a struct, but not actually
part of a field. This most commonly occurs with lifetimes. For instance, the `Iter`
for `&'a [T]` is (approximately) defined as follows:

```
pub struct Iter<'a, T: 'a> {
    ptr: *const T,
    end: *const T,
}
```

However because `'a` is unused within the struct's body, it's *unbound*.
Because of the troubles this has historically caused, unbound lifetimes and
types are *illegal* in struct definitions. Therefore we must somehow refer
to these types in the body. Correctly doing this is necessary to have
correct variance and drop checking.

We do this using *PhantomData*, which is a special marker type. PhantomData
consumes no space, but simulates a field of the given type for the purpose of
static analysis. This was deemed to be less error-prone than explicitly telling
the type-system the kind of variance that you want, while also providing other
useful information.

Iter logically contains `&'a T`, so this is exactly what we tell
the PhantomData to simulate:

```
pub struct Iter<'a, T: 'a> {
    ptr: *const T,
    end: *const T,
    _marker: marker::PhantomData<&'a T>,
}
```




# Dropck

When a type is going out of scope, Rust will try to Drop it. Drop executes
arbitrary code, and in fact allows us to "smuggle" arbitrary code execution
into many places. As such additional soundness checks (dropck) are necessary to
ensure that a type T can be safely instantiated and dropped. It turns out that we
*really* don't need to care about dropck in practice, as it often "just works".

However the one exception is with PhantomData. Given a struct like Vec:

```
struct Vec<T> {
    data: *const T, // *const for variance!
    len: usize,
    cap: usize,
}
```

dropck will generously determine that Vec<T> does not own any values of
type T. This will unfortunately allow people to construct unsound Drop
implementations that access data that has already been dropped. In order to
tell dropck that we *do* own values of type T, and may call destructors of that
type, we must add extra PhantomData:

```
struct Vec<T> {
    data: *const T, // *const for covariance!
    len: usize,
    cap: usize,
    _marker: marker::PhantomData<T>,
}
```

Raw pointers that own an allocation is such a pervasive pattern that the
standard library made a utility for itself called `Unique<T>` which:

* wraps a `*const T`,
* includes a PhantomData<T>,
* auto-derives Send/Sync as if T was contained
* marks the pointer as NonZero for the null-pointer optimization




# Splitting Lifetimes

The mutual exclusion property of mutable references can be very limiting when
working with a composite structure. Borrowck understands some basic stuff, but
will fall over pretty easily. Borrowck understands structs sufficiently to
understand that it's possible to borrow disjoint fields of a struct simultaneously.
So this works today:

```rust
struct Foo {
    a: i32,
    b: i32,
    c: i32,
}

let mut x = Foo {a: 0, b: 0, c: 0};
let a = &mut x.a;
let b = &mut x.b;
let c = &x.c;
*b += 1;
let c2 = &x.c;
*a += 10;
println!("{} {} {} {}", a, b, c, c2);
```

However borrowck doesn't understand arrays or slices in any way, so this doesn't
work:

```rust
let x = [1, 2, 3];
let a = &mut x[0];
let b = &mut x[1];
println!("{} {}", a, b);
```

```text
<anon>:3:18: 3:22 error: cannot borrow immutable indexed content `x[..]` as mutable
<anon>:3     let a = &mut x[0];
                          ^~~~
<anon>:4:18: 4:22 error: cannot borrow immutable indexed content `x[..]` as mutable
<anon>:4     let b = &mut x[1];
                          ^~~~
error: aborting due to 2 previous errors
```

While it was plausible that borrowck could understand this simple case, it's
pretty clearly hopeless for borrowck to understand disjointness in general
container types like a tree, especially if distinct keys actually *do* map
to the same value.

In order to "teach" borrowck that what we're doing is ok, we need to drop down
to unsafe code. For instance, mutable slices expose a `split_at_mut` function that
consumes the slice and returns *two* mutable slices. One for everything to the
left of the index, and one for everything to the right. Intuitively we know this
is safe because the slices don't alias. However the implementation requires some
unsafety:

```rust
fn split_at_mut(&mut self, mid: usize) -> (&mut [T], &mut [T]) {
    unsafe {
        let self2: &mut [T] = mem::transmute_copy(&self);

        (ops::IndexMut::index_mut(self, ops::RangeTo { end: mid } ),
         ops::IndexMut::index_mut(self2, ops::RangeFrom { start: mid } ))
    }
}
```

This is pretty plainly dangerous. We use transmute to duplicate the slice with an
*unbounded* lifetime, so that it can be treated as disjoint from the other until
we unify them when we return.

However more subtle is how iterators that yield mutable references work.
The iterator trait is defined as follows:

```rust
trait Iterator {
    type Item;

    fn next(&mut self) -> Option<Self::Item>;
}
```

Given this definition, Self::Item has *no* connection to `self`. This means
that we can call `next` several times in a row, and hold onto all the results
*concurrently*. This is perfectly fine for by-value iterators, which have exactly
these semantics. It's also actually fine for shared references, as they admit
arbitrarily many references to the same thing (although the
iterator needs to be a separate object from the thing being shared). But mutable
references make this a mess. At first glance, they might seem completely
incompatible with this API, as it would produce multiple mutable references to
the same object!

However it actually *does* work, exactly because iterators are one-shot objects.
Everything an IterMut yields will be yielded *at most* once, so we don't *actually*
ever yield multiple mutable references to the same piece of data.

In general all mutable iterators require *some* unsafe code *somewhere*, though.
Whether it's raw pointers, or safely composing on top of *another* IterMut.

For instance, VecDeque's IterMut:

```rust
pub struct IterMut<'a, T:'a> {
    // The whole backing array. Some of these indices are initialized!
    ring: &'a mut [T],
    tail: usize,
    head: usize,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        if self.tail == self.head {
            return None;
        }
        let tail = self.tail;
        self.tail = wrap_index(self.tail.wrapping_add(1), self.ring.len());

        unsafe {
            // might as well do unchecked indexing since wrap_index has us
            // in-bounds, and many of the "middle" indices are uninitialized
            // anyway.
            let elem = self.ring.get_unchecked_mut(tail);

            // round-trip through a raw pointer to unbound the lifetime from
            // ourselves
            Some(&mut *(elem as *mut _))
        }
    }
}
```

A very subtle but interesting detail in this design is that it *relies on
privacy to be sound*. Borrowck works on some very simple rules. One of those rules
is that if we have a live &mut Foo and Foo contains an &mut Bar, then that &mut
Bar is *also* live. Since IterMut is always live when `next` can be called, if
`ring` were public then we could mutate `ring` while outstanding mutable borrows
to it exist!





# Weird Lifetimes

Given the following code:

```rust
struct Foo;

impl Foo {
    fn mutate_and_share(&mut self) -> &Self { &*self }
    fn share(&self) {}
}

fn main() {
    let mut foo = Foo;
    let loan = foo.mutate_and_share();
    foo.share();
}
```

One might expect it to compile. We call `mutate_and_share`, which mutably borrows
`foo` *temporarily*, but then returns *only* a shared reference. Therefore we
would expect `foo.share()` to succeed as `foo` shouldn't be mutably borrowed.

However when we try to compile it:

```text
<anon>:11:5: 11:8 error: cannot borrow `foo` as immutable because it is also borrowed as mutable
<anon>:11     foo.share();
              ^~~
<anon>:10:16: 10:19 note: previous borrow of `foo` occurs here; the mutable borrow prevents subsequent moves, borrows, or modification of `foo` until the borrow ends
<anon>:10     let loan = foo.mutate_and_share();
                         ^~~
<anon>:12:2: 12:2 note: previous borrow ends here
<anon>:8 fn main() {
<anon>:9     let mut foo = Foo;
<anon>:10     let loan = foo.mutate_and_share();
<anon>:11     foo.share();
<anon>:12 }
          ^
```

What happened? Well, the lifetime of `loan` is derived from a *mutable* borrow.
This makes the type system believe that `foo` is mutably borrowed as long as
`loan` exists, even though it's a shared reference. This isn't a bug, although
one could argue it is a limitation of the design. In particular, to know if
the mutable part of the borrow is *really* expired we'd have to peek into
implementation details of the function. Currently, type-checking a function
does not need to inspect the bodies of any other functions or types.


