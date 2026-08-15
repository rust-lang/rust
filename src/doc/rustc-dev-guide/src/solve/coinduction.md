# Coinduction

The trait solver may use coinduction when proving goals.
Coinduction is fairly subtle so we're giving it its own chapter.

## Coinduction and induction

With induction, we recursively apply proofs until we end up with a finite proof tree.
Consider the example of `Vec<Vec<Vec<u32>>>: Debug` which results in the following tree.

- `Vec<Vec<Vec<u32>>>: Debug`
    - `Vec<Vec<u32>>: Debug`
        - `Vec<u32>: Debug`
            - `u32: Debug`

This tree is finite. But not all goals we would want to hold have finite proof trees,
consider the following example:

```rust
struct List<T> {
    value: T,
    next: Option<Box<List<T>>>,
}
```

For `List<T>: Send` to hold all its fields have to recursively implement `Send` as well.
This would result in the following proof tree:

- `List<T>: Send`
    - `T: Send`
    - `Option<Box<List<T>>>: Send`
        - `Box<List<T>>: Send`
            - `List<T>: Send`
                - `T: Send`
                - `Option<Box<List<T>>>: Send`
                    - `Box<List<T>>: Send`
                        - ...

This tree would be infinitely large which is exactly what coinduction is about. 

> To **inductively** prove a goal you need to provide a finite proof tree for it.
> To **coinductively** prove a goal the provided proof tree may be infinite.

## Why is coinduction correct

When checking whether some trait goals holds, we're asking "does there exist an `impl`
which satisfies this bound". Even if are infinite chains of nested goals, we still have a
unique `impl` which should be used.

## How to implement coinduction

While our implementation can not check for coinduction by trying to construct an infinite
tree as that would take infinite resources, it still makes sense to think of coinduction
from this perspective.

As we cannot check for infinite trees, we instead search for patterns for which we know that
they would result in an infinite proof tree. The currently pattern we detect are (canonical)
cycles. If `T: Send` relies on `T: Send` then it's pretty clear that this will just go on forever.

With cycles we have to be careful with caching. Because of canonicalization of regions and
inference variables encountering a cycle doesn't mean that we would get an infinite proof tree.
Looking at the following example:
```rust
trait Foo {}
struct Wrapper<T>(T);

impl<T> Foo for Wrapper<Wrapper<T>>
where
    Wrapper<T>: Foo
{} 
```
Proving `Wrapper<?0>: Foo` uses the impl `impl<T> Foo for Wrapper<Wrapper<T>>` which constrains
`?0` to `Wrapper<?1>` and then requires `Wrapper<?1>: Foo`. Due to canonicalization this would be
detected as a cycle.

The idea to solve is to return a *provisional result* whenever we detect a cycle and repeatedly
retry goals until the *provisional result* is equal to the final result of that goal. We
start out by using `Yes` with no constraints as the result and then update it to the result of
the previous iteration whenever we have to rerun.

TODO: elaborate here. We use the same approach as chalk for coinductive cycles.
Note that the treatment for inductive cycles currently differs by simply returning `Overflow`.
See [the relevant chapters][chalk] in the chalk book.

[chalk]: https://rust-lang.github.io/chalk/book/recursive/inductive_cycles.html


## Future work

We currently only consider auto-traits, `Sized`, and `WF`-goals to be coinductive.
In the future we pretty much intend for all goals to be coinductive.
Lets first elaborate on why allowing more coinductive proofs is even desirable.

### Recursive data types already rely on coinduction...

...they just tend to avoid them in the trait solver.

```rust
enum List<T> {
    Nil,
    Succ(T, Box<List<T>>),
}

impl<T: Clone> Clone for List<T> {
    fn clone(&self) -> Self {
        match self {
            List::Nil => List::Nil,
            List::Succ(head, tail) => List::Succ(head.clone(), tail.clone()),
        }
    }
}
```

We are using `tail.clone()` in this impl. For this we have to prove `Box<List<T>>: Clone`
which requires `List<T>: Clone` but that relies on the impl which we are currently checking.
By adding that requirement to the `where`-clauses of the impl, which is what we would
do with [perfect derive], we move that cycle into the trait solver and [get an error][ex1].

### Recursive data types

We also need coinduction to reason about recursive types containing projections,
e.g. the following currently fails to compile even though it should be valid.
```rust
use std::borrow::Cow;
pub struct Foo<'a>(Cow<'a, [Foo<'a>]>);
```
This issue has been known since at least 2015, see
[#23714](https://github.com/rust-lang/rust/issues/23714) if you want to know more.

### Explicitly checked implied bounds

When checking an impl, we assume that the types in the impl headers are well-formed.
This means that when using instantiating the impl we have to prove that's actually the case.
[#100051](https://github.com/rust-lang/rust/issues/100051) shows that this is not the case.
To fix this, we have to add `WF` predicates for the types in impl headers.
Without coinduction for all traits, this even breaks `core`.

```rust
trait FromResidual<R> {}
trait Try: FromResidual<<Self as Try>::Residual> {
    type Residual;
}

struct Ready<T>(T);
impl<T> Try for Ready<T> {
    type Residual = Ready<()>;
}
impl<T> FromResidual<<Ready<T> as Try>::Residual> for Ready<T> {}
```

When checking that the impl of `FromResidual` is well formed we get the following cycle:

The impl is well formed if `<Ready<T> as Try>::Residual` and `Ready<T>` are well formed.
- `wf(<Ready<T> as Try>::Residual)` requires
-  `Ready<T>: Try`, which requires because of the super trait
-  `Ready<T>: FromResidual<Ready<T> as Try>::Residual>`, **because of implied bounds on impl**
-  `wf(<Ready<T> as Try>::Residual)` :tada: **cycle**

### Issues when extending coinduction to more goals

There are some additional issues to keep in mind when extending coinduction.
The issues here are not relevant for the current solver.

#### Implied super trait bounds

Our trait system currently treats super traits, e.g. `trait Trait: SuperTrait`,
by 1) requiring that `SuperTrait` has to hold for all types which implement `Trait`,
and 2) assuming `SuperTrait` holds if `Trait` holds.

Relying on 2) while proving 1) is unsound. This can only be observed in case of
coinductive cycles. Without cycles, whenever we rely on 2) we must have also
proven 1) without relying on 2) for the used impl of `Trait`.

```rust
trait Trait: SuperTrait {}

impl<T: Trait> Trait for T {}

// Keeping the current setup for coinduction
// would allow this compile. Uff :<
fn sup<T: SuperTrait>() {}
fn requires_trait<T: Trait>() { sup::<T>() }
fn generic<T>() { requires_trait::<T>() }
```
This is not really fundamental to coinduction but rather an existing property
which is made unsound because of it.

##### Possible solutions

The easiest way to solve this would be to completely remove 2) and always elaborate
`T: Trait` to `T: Trait` and `T: SuperTrait` outside of the trait solver.
This would allow us to also remove 1), but as we still have to prove ordinary
`where`-bounds on traits, that's just additional work.

While one could imagine ways to disable cyclic uses of 2) when checking 1),
at least the ideas of myself - @lcnr - are all far to complex to be reasonable.

#### `normalizes_to` goals and progress

A `normalizes_to` goal represents the requirement that `<T as Trait>::Assoc` normalizes
to some `U`. This is achieved by defacto first normalizing `<T as Trait>::Assoc` and then
equating the resulting type with `U`. It should be a mapping as each projection should normalize
to exactly one type. By simply allowing infinite proof trees, we would get the following behavior:

```rust
trait Trait {
    type Assoc;
}

impl Trait for () {
    type Assoc = <() as Trait>::Assoc;
}
```

If we now compute `normalizes_to(<() as Trait>::Assoc, Vec<u32>)`, we would resolve the impl
and get the associated type `<() as Trait>::Assoc`. We then equate that with the expected type,
causing us to check `normalizes_to(<() as Trait>::Assoc, Vec<u32>)` again.
This just goes on forever, resulting in an infinite proof tree.

This means that `<() as Trait>::Assoc` would be equal to any other type which is unsound.

##### How to solve this

**WARNING: THIS IS SUBTLE AND MIGHT BE WRONG**

Unlike trait goals, `normalizes_to` has to be *productive*[^1]. A `normalizes_to` goal
is productive once the projection normalizes to a rigid type constructor,
so `<() as Trait>::Assoc` normalizing to `Vec<<() as Trait>::Assoc>` would be productive.

A `normalizes_to` goal has two kinds of nested goals. Nested requirements needed to actually
normalize the projection, and the equality between the normalized projection and the
expected type. Only the equality has to be productive. A branch in the proof tree is productive
if it is either finite, or contains at least one `normalizes_to` where the alias is resolved
to a rigid type constructor.

Alternatively, we could simply always treat the equate branch of `normalizes_to` as inductive.
Any cycles should result in infinite types, which aren't supported anyways and would only
result in overflow when deeply normalizing for codegen.

experimentation and examples: <https://hackmd.io/-8p0AHnzSq2VAE6HE_wX-w?view>

Another attempt at a summary.
- in projection eq, we must make progress with constraining the rhs
- a cycle is only ok if while equating we have a rigid ty on the lhs after norm at least once
- cycles outside of the recursive `eq` call of `normalizes_to` are always fine

[^1]: related: <https://coq.inria.fr/refman/language/core/coinductive.html#top-level-definitions-of-corecursive-functions>

[perfect derive]: https://smallcultfollowing.com/babysteps/blog/2022/04/12/implied-bounds-and-perfect-derive
[ex1]: https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=0a9c3830b93a2380e6978d6328df8f72
