# Implied Bounds

Implied bounds remove the need to repeat where clauses written on
a type declaration or a trait declaration. For example, say we have the
following type declaration:
```rust,ignore
struct HashSet<K: Hash> {
    ...
}
```

then everywhere we use `HashSet<K>` as an "input" type, that is appearing in
the receiver type of an `impl` or in the arguments of a function, we don't
want to have to repeat the `where K: Hash` bound, as in:

```rust,ignore
// I don't want to have to repeat `where K: Hash` here.
impl<K> HashSet<K> {
    ...
}

// Same here.
fn loud_insert<K>(set: &mut HashSet<K>, item: K) {
    println!("inserting!");
    set.insert(item);
}
```

Note that in the `loud_insert` example, `HashSet<K>` is not the type
of the `set` argument of `loud_insert`, it only *appears* in the
argument type `&mut HashSet<K>`: we care about every type appearing
in the function's header (the header is the signature without the return type),
not only types of the function's arguments.

The rationale for applying implied bounds to input types is that, for example,
in order to call the `loud_insert` function above, the programmer must have
*produced* the type `HashSet<K>` already, hence the compiler already verified
that `HashSet<K>` was well-formed, i.e. that `K` effectively implemented
`Hash`, as in the following example:

```rust,ignore
fn main() {
    // I am producing a value of type `HashSet<i32>`.
    // If `i32` was not `Hash`, the compiler would report an error here.
    let set: HashSet<i32> = HashSet::new();
    loud_insert(&mut set, 5);
}
```

Hence, we don't want to repeat where clauses for input types because that would
sort of duplicate the work of the programmer, having to verify that their types
are well-formed both when calling the function and when using them in the
arguments of their function. The same reasoning applies when using an `impl`.

Similarly, given the following trait declaration:
```rust,ignore
trait Copy where Self: Clone { // desugared version of `Copy: Clone`
    ...
}
```

then everywhere we bound over `SomeType: Copy`, we would like to be able to
use the fact that `SomeType: Clone` without having to write it explicitly,
as in:
```rust,ignore
fn loud_clone<T: Clone>(x: T) {
    println!("cloning!");
    x.clone();
}

fn fun_with_copy<T: Copy>(x: T) {
    println!("will clone a `Copy` type soon...");

    // I'm using `loud_clone<T: Clone>` with `T: Copy`, I know this
    // implies `T: Clone` so I don't want to have to write it explicitly.
    loud_clone(x);
}
```

The rationale for implied bounds for traits is that if a type implements
`Copy`, that is, if there exists an `impl Copy` for that type, there *ought*
to exist an `impl Clone` for that type, otherwise the compiler would have
reported an error in the first place. So again, if we were forced to repeat the
additionnal `where SomeType: Clone` everywhere whereas we already know that
`SomeType: Copy` hold, we would kind of duplicate the verification work.

Implied bounds are not yet completely enforced in rustc, at the moment it only
works for outlive requirements, super trait bounds, and bounds on associated
types. The full RFC can be found [here][RFC]. We'll give here a brief view
of how implied bounds work and why we chose to implement it that way. The
complete set of lowering rules can be found in the corresponding
[chapter](./lowering-rules.md).

[RFC]: https://github.com/rust-lang/rfcs/blob/master/text/2089-implied-bounds.md

## Implied bounds and lowering rules

Now we need to express implied bounds in terms of logical rules. We will start
with exposing a naive way to do it. Suppose that we have the following traits:
```rust,ignore
trait Foo {
    ...
}

trait Bar where Self: Foo { } {
    ...
}
```

So we would like to say that if a type implements `Bar`, then necessarily
it must also implement `Foo`. We might think that a clause like this would
work:
```text
forall<Type> {
    Implemented(Type: Foo) :- Implemented(Type: Bar).
}
```

Now suppose that we just write this impl:
```rust,ignore
struct X;

impl Bar for X { }
```

Clearly this should not be allowed: indeed, we wrote a `Bar` impl for `X`, but
the `Bar` trait requires that we also implement `Foo` for `X`, which we never
did. In terms of what the compiler does, this would look like this:
```rust,ignore
struct X;

impl Bar for X {
    // We are in a `Bar` impl for the type `X`.
    // There is a `where Self: Foo` bound on the `Bar` trait declaration.
    // Hence I need to prove that `X` also implements `Foo` for that impl
    // to be legal.
}
```
So the compiler would try to prove `Implemented(X: Foo)`. Of course it will
not find any `impl Foo for X` since we did not write any. However, it
will see our implied bound clause:
```text
forall<Type> {
    Implemented(Type: Foo) :- Implemented(Type: Bar).
}
```

so that it may be able to prove `Implemented(X: Foo)` if `Implemented(X: Bar)`
holds. And it turns out that `Implemented(X: Bar)` does hold since we wrote
a `Bar` impl for `X`! Hence the compiler will accept the `Bar` impl while it
should not.

## Implied bounds coming from the environment

So the naive approach does not work. What we need to do is to somehow decouple
implied bounds from impls. Suppose we know that a type `SomeType<...>`
implements `Bar` and we want to deduce that `SomeType<...>` must also implement
`Foo`.

There are two possibilities: first, we have enough information about
`SomeType<...>` to see that there exists a `Bar` impl in the program which
covers `SomeType<...>`, for example a plain `impl<...> Bar for SomeType<...>`.
Then if the compiler has done its job correctly, there *must* exist a `Foo`
impl which covers `SomeType<...>`, e.g. another plain
`impl<...> Foo for SomeType<...>`. In that case then, we can just use this
impl and we do not need implied bounds at all.

Second possibility: we do not know enough about `SomeType<...>` in order to
find a `Bar` impl which covers it, for example if `SomeType<...>` is just
a type parameter in a function:
```rust,ignore
fn foo<T: Bar>() {
    // We'd like to deduce `Implemented(T: Foo)`.
}
```

That is, the information that `T` implements `Bar` here comes from the
*environment*. The environment is the set of things that we assume to be true
when we type check some Rust declaration. In that case, what we assume is that
`T: Bar`. Then at that point, we might authorize ourselves to have some kind
of  "local" implied bound reasoning which would say
`Implemented(T: Foo) :- Implemented(T: Bar)`. This reasoning would
only be done within our `foo` function in order to avoid the earlier
problem where we had a global clause.

We can apply these local reasonings everywhere we can have an environment
-- i.e. when we can write where clauses -- that is, inside impls,
trait declarations, and type declarations.

## Computing implied bounds with `FromEnv`

The previous subsection showed that it was only useful to compute implied
bounds for facts coming from the environment.
We talked about "local" rules, but there are multiple possible strategies to
indeed implement the locality of implied bounds.

In rustc, the current strategy is to *elaborate* bounds: that is, each time
we have a fact in the environment, we recursively derive all the other things
that are implied by this fact until we reach a fixed point. For example, if
we have the following declarations:
```rust,ignore
trait A { }
trait B where Self: A { }
trait C where Self: B { }

fn foo<T: C>() {
    ...
}
```
then inside the `foo` function, we start with an environment containing only
`Implemented(T: C)`. Then because of implied bounds for the `C` trait, we
elaborate `Implemented(T: B)` and add it to our environment. Because of
implied bounds for the `B` trait, we elaborate `Implemented(T: A)`and add it
to our environment as well. We cannot elaborate anything else, so we conclude
that our final environment consists of `Implemented(T: A + B + C)`.

In the new-style trait system, we like to encode as many things as possible
with logical rules. So rather than "elaborating", we have a set of *global*
program clauses defined like so:
```text
forall<T> { Implemented(T: A) :- FromEnv(T: A). }

forall<T> { Implemented(T: B) :- FromEnv(T: B). }
forall<T> { FromEnv(T: A) :- FromEnv(T: B). }

forall<T> { Implemented(T: C) :- FromEnv(T: C). }
forall<T> { FromEnv(T: C) :- FromEnv(T: C). }
```
So these clauses are defined globally (that is, they are available from
everywhere in the program) but they cannot be used because the hypothesis
is always of the form `FromEnv(...)` which is a bit special. Indeed, as
indicated by the name, `FromEnv(...)` facts can **only** come from the
environment.
How it works is that in the `foo` function, instead of having an environment
containing `Implemented(T: C)`, we replace this environment with
`FromEnv(T: C)`. From here and thanks to the above clauses, we see that we
are able to reach any of `Implemented(T: A)`, `Implemented(T: B)` or
`Implemented(T: C)`, which is what we wanted.

## Implied bounds and well-formedness checking

Implied bounds are tightly related with well-formedness checking.
Well-formedness checking is the process of checking that the impls the
programmer wrote are legal, what we referred to earlier as "the compiler doing
its job correctly".

We already saw examples of illegal and legal impls:
```rust,ignore
trait Foo { }
trait Bar where Self: Foo { }

struct X;
struct Y;

impl Bar for X {
    // This impl is not legal: the `Bar` trait requires that we also
    // implement `Foo`, and we didn't.
}

impl Foo for Y {
    // This impl is legal: there is nothing to check as there are no where
    // clauses on the `Foo` trait.
}

impl Bar for Y {
    // This impl is legal: we have a `Foo` impl for `Y`.
}
```
We must define what "legal" and "illegal" mean. For this, we introduce another
predicate: `WellFormed(Type: Trait)`. We say that the trait reference
`Type: Trait` is well-formed if `Type` meets the bounds written on the
`Trait` declaration. For each impl we write, assuming that the where clauses
declared on the impl hold, the compiler tries to prove that the corresponding
trait reference is well-formed. The impl is legal if the compiler manages to do
so.

Coming to the definition of `WellFormed(Type: Trait)`, it would be tempting
to define it as:
```rust,ignore
trait Trait where WC1, WC2, ..., WCn {
    ...
}
```

```text
forall<Type> {
    WellFormed(Type: Trait) :- WC1 && WC2 && .. && WCn.
}
```
and indeed this was basically what was done in rustc until it was noticed that
this mixed badly with implied bounds. The key thing is that implied bounds
allows someone to derive all bounds implied by a fact in the environment, and
this *transitively* as we've seen with the `A + B + C` traits example.
However, the `WellFormed` predicate as defined above only checks that the
*direct* superbounds hold. That is, if we come back to our `A + B + C`
example:
```rust,ignore
trait A { }
// No where clauses, always well-formed.
// forall<Type> { WellFormed(Type: A). }

trait B where Self: A { }
// We only check the direct superbound `Self: A`.
// forall<Type> { WellFormed(Type: B) :- Implemented(Type: A). }

trait C where Self: B { }
// We only check the direct superbound `Self: B`. We do not check
// the `Self: A` implied bound  coming from the `Self: B` superbound.
// forall<Type> { WellFormed(Type: C) :- Implemented(Type: B). }
```
There is an asymmetry between the recursive power of implied bounds and
the shallow checking of `WellFormed`. It turns out that this asymmetry
can be [exploited][bug]. Indeed, suppose that we define the following
traits:
```rust,ignore
trait Partial where Self: Copy { }
// WellFormed(Self: Partial) :- Implemented(Self: Copy).

trait Complete where Self: Partial { }
// WellFormed(Self: Complete) :- Implemented(Self: Partial).

impl<T> Partial for T where T: Complete { }

impl<T> Complete for T { }
```

For the `Partial` impl, what the compiler must prove is:
```text
forall<T> {
    if (T: Complete) { // assume that the where clauses hold
        WellFormed(T: Partial) // show that the trait reference is well-formed
    }
}
```
Proving `WellFormed(T: Partial)` amounts to proving `Implemented(T: Copy)`.
However, we have `Implemented(T: Complete)` in our environment: thanks to
implied bounds, we can deduce `Implemented(T: Partial)`. Using implied bounds
one level deeper, we can deduce `Implemented(T: Copy)`. Finally, the `Partial`
impl is legal.

For the `Complete` impl, what the compiler must prove is:
```text
forall<T> {
    WellFormed(T: Complete) // show that the trait reference is well-formed
}
```
Proving `WellFormed(T: Complete)` amounts to proving `Implemented(T: Partial)`.
We see that the `impl Partial for T` applies if we can prove
`Implemented(T: Complete)`, and it turns out we can prove this fact since our
`impl<T> Complete for T` is a blanket impl without any where clauses.

So both impls are legal and the compiler accepts the program. Moreover, thanks
to the `Complete` blanket impl, all types implement `Complete`. So we could
now use this impl like so:
```rust,ignore
fn eat<T>(x: T) { }

fn copy_everything<T: Complete>(x: T) {
    eat(x);
    eat(x);
}

fn main() {
    let not_copiable = vec![1, 2, 3, 4];
    copy_everything(not_copiable);
}
```
In this program, we use the fact that `Vec<i32>` implements `Complete`, as any
other type. Hence we can call `copy_everything` with an argument of type
`Vec<i32>`. Inside the `copy_everything` function, we have the
`Implemented(T: Complete)` bound in our environment. Thanks to implied bounds,
we can deduce `Implemented(T: Partial)`. Using implied bounds again, we deduce
`Implemented(T: Copy)` and we can indeed call the `eat` function which moves
the argument twice since its argument is `Copy`. Problem: the `T` type was
in fact `Vec<i32>` which is not copy at all, hence we will double-free the
underlying vec storage so we have a memory unsoundness in safe Rust.

Of course, disregarding the asymmetry between `WellFormed` and implied bounds,
this bug was possible only because we had some kind of self-referencing impls.
But self-referencing impls are very useful in practice and are not the real
culprits in this affair.

[bug]: https://github.com/rust-lang/rust/pull/43786

## Co-inductiveness of `WellFormed`

So the solution is to fix this asymmetry between `WellFormed` and implied
bounds. For that, we need for the `WellFormed` predicate to not only require
that the direct superbounds hold, but also all the bounds transitively implied
by the superbounds. What we can do is to have the following rules for the
`WellFormed` predicate:
```rust,ignore
trait A { }
// WellFormed(Self: A) :- Implemented(Self: A).

trait B where Self: A { }
// WellFormed(Self: B) :- Implemented(Self: B) && WellFormed(Self: A).

trait C where Self: B { }
// WellFormed(Self: C) :- Implemented(Self: C) && WellFormed(Self: B).
```

Notice that we are now also requiring `Implemented(Self: Trait)` for
`WellFormed(Self: Trait)` to be true: this is to simplify the process of
traversing all the implied bounds transitively. This does not change anything
when checking whether impls are legal, because since we assume
that the where clauses hold inside the impl, we know that the corresponding
trait reference do hold. Thanks to this setup, you can see that we indeed
require to prove the set of all bounds transitively implied by the where
clauses.

However there is still a catch. Suppose that we have the following trait
definition:
```rust,ignore
trait Foo where <Self as Foo>::Item: Foo {
    type Item;
}
```

so this definition is a bit more involved than the ones we've seen already
because it defines an associated item. However, the well-formedness rule
would not be more complicated:
```text
WellFormed(Self: Foo) :-
    Implemented(Self: Foo) &&
    WellFormed(<Self as Foo>::Item: Foo).
```

Now we would like to write the following impl:
```rust,ignore
impl Foo for i32 {
    type Item = i32;
}
```
The `Foo` trait definition and the `impl Foo for i32` are perfectly valid
Rust: we're kind of recursively using our `Foo` impl in order to show that
the associated value indeed implements `Foo`, but that's ok. But if we
translate this to our well-formedness setting, the compiler proof process
inside the `Foo` impl is the following: it starts with proving that the
well-formedness goal `WellFormed(i32: Foo)` is true. In order to do that,
it must prove the following goals: `Implemented(i32: Foo)` and
`WellFormed(<i32 as Foo>::Item: Foo)`. `Implemented(i32: Foo)` holds because
there is our impl and there are no where clauses on it so it's always true.
However, because of the associated type value we used,
`WellFormed(<i32 as Foo>::Item: Foo)` simplifies to just
`WellFormed(i32: Foo)`. So in order to prove its original goal
`WellFormed(i32: Foo)`, the compiler needs to prove `WellFormed(i32: Foo)`:
this clearly is a cycle and cycles are usually rejected by the trait solver,
unless...  if the `WellFormed` predicate was made to be co-inductive.

A co-inductive predicate, as discussed in the chapter on
[goals and clauses](./goals-and-clauses.md#coinductive-goals), are predicates
for which the
trait solver accepts cycles. In our setting, this would be a valid thing to do:
indeed, the `WellFormed` predicate just serves as a way of enumerating all
the implied bounds. Hence, it's like a fixed point algorithm: it tries to grow
the set of implied bounds until there is nothing more to add. Here, a cycle
in the chain of `WellFormed` predicates just means that there is no more bounds
to add in that direction, so we can just accept this cycle and focus on other
directions. It's easy to prove that under these co-inductive semantics, we
are effectively visiting all the transitive implied bounds, and only these.

## Implied bounds on types

We mainly talked about implied bounds for traits because this was the most
subtle regarding implementation. Implied bounds on types are simpler,
especially because if we assume that a type is well-formed, we don't use that
fact to deduce that other types are well-formed, we only use it to deduce
that e.g. some trait bounds hold.

For types, we just use rules like these ones:
```rust,ignore
struct Type<...> where WC1, ..., WCn {
    ...
}
```

```text
forall<...> {
    WellFormed(Type<...>) :- WC1, ..., WCn.
}

forall<...> {
    FromEnv(WC1) :- FromEnv(Type<...>).
    ...
    FromEnv(WCn) :- FromEnv(Type<...>).
}
```
We can see that we have this asymmetry between well-formedness check,
which only verifies that the direct superbounds hold, and implied bounds which
gives access to all bounds transitively implied by the where clauses. In that
case this is ok because as we said, we don't use `FromEnv(Type<...>)` to deduce
other `FromEnv(OtherType<...>)` things, nor do we use `FromEnv(Type: Trait)` to
deduce `FromEnv(OtherType<...>)` things. So in that sense type definitions are
"less recursive" than traits, and we saw in a previous subsection that
it was the combination of asymmetry and recursive trait / impls that led to
unsoundness. As such, the `WellFormed(Type<...>)` predicate does not need
to be co-inductive.

This asymmetry optimization is useful because in a real Rust program, we have
to check the well-formedness of types very often (e.g. for each type which
appears in the body of a function).
