# Canonical queries

The "start" of the trait system is the **canonical query** (these are
both queries in the more general sense of the word – something you
would like to know the answer to – and in the
[rustc-specific sense](../query.html)).  The idea is that the type
checker or other parts of the system, may in the course of doing their
thing want to know whether some trait is implemented for some type
(e.g., is `u32: Debug` true?). Or they may want to
normalize some associated type.

This section covers queries at a fairly high level of abstraction. The
subsections look a bit more closely at how these ideas are implemented
in rustc.

## The traditional, interactive Prolog query

In a traditional Prolog system, when you start a query, the solver
will run off and start supplying you with every possible answer it can
find. So given something like this:

```text
?- Vec<i32>: AsRef<?U>
```

The solver might answer:

```text
Vec<i32>: AsRef<[i32]>
    continue? (y/n)
```

This `continue` bit is interesting. The idea in Prolog is that the
solver is finding **all possible** instantiations of your query that
are true. In this case, if we instantiate `?U = [i32]`, then the query
is true (note that a traditional Prolog interface does not, directly,
tell us a value for `?U`, but we can infer one by unifying the
response with our original query – Rust's solver gives back a
substitution instead). If we were to hit `y`, the solver might then
give us another possible answer:

```text
Vec<i32>: AsRef<Vec<i32>>
    continue? (y/n)
```

This answer derives from the fact that there is a reflexive impl
(`impl<T> AsRef<T> for T`) for `AsRef`. If were to hit `y` again,
then we might get back a negative response:

```text
no
```

Naturally, in some cases, there may be no possible answers, and hence
the solver will just give me back `no` right away:

```text
?- Box<i32>: Copy
    no
```

In some cases, there might be an infinite number of responses. So for
example if I gave this query, and I kept hitting `y`, then the solver
would never stop giving me back answers:

```text
?- Vec<?U>: Clone
    Vec<i32>: Clone
        continue? (y/n)
    Vec<Box<i32>>: Clone
        continue? (y/n)
    Vec<Box<Box<i32>>>: Clone
        continue? (y/n)
    Vec<Box<Box<Box<i32>>>>: Clone
        continue? (y/n)
```

As you can imagine, the solver will gleefully keep adding another
layer of `Box` until we ask it to stop, or it runs out of memory.

Another interesting thing is that queries might still have variables
in them. For example:

```text
?- Rc<?T>: Clone
```

might produce the answer:

```text
Rc<?T>: Clone
    continue? (y/n)
```

After all, `Rc<?T>` is true **no matter what type `?T` is**.

<a id="query-response"></a>

## A trait query in rustc

The trait queries in rustc work somewhat differently. Instead of
trying to enumerate **all possible** answers for you, they are looking
for an **unambiguous** answer. In particular, when they tell you the
value for a type variable, that means that this is the **only possible
instantiation** that you could use, given the current set of impls and
where-clauses, that would be provable.

The response to a trait query in rustc is typically a
`Result<QueryResult<T>, NoSolution>` (where the `T` will vary a bit
depending on the query itself). The `Err(NoSolution)` case indicates
that the query was false and had no answers (e.g., `Box<i32>: Copy`).
Otherwise, the `QueryResult` gives back information about the possible answer(s)
we did find. It consists of four parts:

- **Certainty:** tells you how sure we are of this answer. It can have two
  values:
  - `Proven` means that the result is known to be true.
    - This might be the result for trying to prove `Vec<i32>: Clone`,
      say, or `Rc<?T>: Clone`.
  - `Ambiguous` means that there were things we could not yet prove to
    be either true *or* false, typically because more type information
    was needed. (We'll see an example shortly.)
    - This might be the result for trying to prove `Vec<?T>: Clone`.
- **Var values:** Values for each of the unbound inference variables
  (like `?T`) that appeared in your original query. (Remember that in Prolog,
  we had to infer these.)
  - As we'll see in the example below, we can get back var values even
    for `Ambiguous` cases.
- **Region constraints:** these are relations that must hold between
  the lifetimes that you supplied as inputs. We'll ignore these here.
- **Value:** The query result also comes with a value of type `T`. For
  some specialized queries – like normalizing associated types –
  this is used to carry back an extra result, but it's often just
  `()`.

### Examples

Let's work through an example query to see what all the parts mean.
Consider [the `Borrow` trait][borrow]. This trait has a number of
impls; among them, there are these two (for clarity, I've written the
`Sized` bounds explicitly):

[borrow]: https://doc.rust-lang.org/std/borrow/trait.Borrow.html

```rust,ignore
impl<T> Borrow<T> for T where T: ?Sized
impl<T> Borrow<[T]> for Vec<T> where T: Sized
```

**Example 1.** Imagine we are type-checking this (rather artificial)
bit of code:

```rust,ignore
fn foo<A, B>(a: A, vec_b: Option<B>) where A: Borrow<B> { }

fn main() {
    let mut t: Vec<_> = vec![]; // Type: Vec<?T>
    let mut u: Option<_> = None; // Type: Option<?U>
    foo(t, u); // Example 1: requires `Vec<?T>: Borrow<?U>`
    ...
}
```

As the comments indicate, we first create two variables `t` and `u`;
`t` is an empty vector and `u` is a `None` option. Both of these
variables have unbound inference variables in their type: `?T`
represents the elements in the vector `t` and `?U` represents the
value stored in the option `u`.  Next, we invoke `foo`; comparing the
signature of `foo` to its arguments, we wind up with `A = Vec<?T>` and
`B = ?U`. Therefore, the where clause on `foo` requires that `Vec<?T>:
Borrow<?U>`. This is thus our first example trait query.

There are many possible solutions to the query `Vec<?T>: Borrow<?U>`;
for example:

- `?U = Vec<?T>`,
- `?U = [?T]`,
- `?T = u32, ?U = [u32]`
- and so forth.

Therefore, the result we get back would be as follows (I'm going to
ignore region constraints and the "value"):

- Certainty: `Ambiguous` – we're not sure yet if this holds
- Var values: `[?T = ?T, ?U = ?U]` – we learned nothing about the values of
  the variables

In short, the query result says that it is too soon to say much about
whether this trait is proven. During type-checking, this is not an
immediate error: instead, the type checker would hold on to this
requirement (`Vec<?T>: Borrow<?U>`) and wait. As we'll see in the next
example, it may happen that `?T` and `?U` wind up constrained from
other sources, in which case we can try the trait query again.

**Example 2.** We can now extend our previous example a bit,
and assign a value to `u`:

```rust,ignore
fn foo<A, B>(a: A, vec_b: Option<B>) where A: Borrow<B> { }

fn main() {
    // What we saw before:
    let mut t: Vec<_> = vec![]; // Type: Vec<?T>
    let mut u: Option<_> = None; // Type: Option<?U>
    foo(t, u); // `Vec<?T>: Borrow<?U>` => ambiguous

    // New stuff:
    u = Some(vec![]); // ?U = Vec<?V>
}
```

As a result of this assignment, the type of `u` is forced to be
`Option<Vec<?V>>`, where `?V` represents the element type of the
vector. This in turn implies that `?U` is [unified] to `Vec<?V>`.

[unified]: ../type-checking.html

Let's suppose that the type checker decides to revisit the
"as-yet-unproven" trait obligation we saw before, `Vec<?T>:
Borrow<?U>`. `?U` is no longer an unbound inference variable; it now
has a value, `Vec<?V>`. So, if we "refresh" the query with that value, we get:

```text
Vec<?T>: Borrow<Vec<?V>>
```

This time, there is only one impl that applies, the reflexive impl:

```text
impl<T> Borrow<T> for T where T: ?Sized
```

Therefore, the trait checker will answer:

- Certainty: `Proven`
- Var values: `[?T = ?T, ?V = ?T]`

Here, it is saying that we have indeed proven that the obligation
holds, and we also know that `?T` and `?V` are the same type (but we
don't know what that type is yet!).

(In fact, as the function ends here, the type checker would give an
error at this point, since the element types of `t` and `u` are still
not yet known, even though they are known to be the same.)


