# [`CoerceUnsized`](https://doc.rust-lang.org/std/ops/trait.CoerceUnsized.html)

`CoerceUnsized` is primarily concerned with data containers. When a struct
(typically, a smart pointer) implements `CoerceUnsized`, that means that the
data it points to is being unsized.

Some implementors of `CoerceUnsized` include:
* `&T`
* `Arc<T>`
* `Box<T>`

This trait is (eventually) intended to be implemented by user-written smart
pointers, and there are rules about when a type is allowed to implement
`CoerceUnsized` that are explained in the trait's documentation.

# [`Unsize`](https://doc.rust-lang.org/std/marker/trait.Unsize.html)

To contrast, the `Unsize` trait is concerned the actual types that are allowed
to be unsized. 

This is not intended to be implemented by users ever, since `Unsize` does not
instruct the compiler (namely codegen) *how* to unsize a type, just whether it
is allowed to be unsized. This is paired somewhat intimately with codegen
which must understand how types are represented and unsized.

## Primitive unsizing implementations

Built-in implementations are provided for:
* `T` -> `dyn Trait + 'a` when `T: Trait` (and `T: Sized + 'a`, and `Trait`
  is object safe).
* `[T; N]` -> `[T]`

## Structural implementations

There are two implementations of `Unsize` which can be thought of as
structural:
* `(A1, A2, .., An): Unsize<(A1, A2, .., U)>` given `An: Unsize<U>`, which
  allows the tail field of a tuple to be unsized. This is gated behind the
  [`unsized_tuple_coercion`] feature.
* `Struct<.., Pi, .., Pj, ..>: Unsize<Struct<.., Ui, .., Uj, ..>>` given 
  `TailField<Pi, .., Pj>: Unsize<Ui, .. Uj>`, which allows the tail field of a
  struct to be unsized if it is the only field that mentions generic parameters
  `Pi`, .., `Pj` (which don't need to be contiguous).

The rules for the latter implementation are slightly complicated, since they
may allow more than one parameter to be changed (not necessarily unsized) and
are best stated in terms of the tail field of the struct.

[`unsized_tuple_coercion`]: https://doc.rust-lang.org/beta/unstable-book/language-features/unsized-tuple-coercion.html

## Upcasting implementations

Two things are called "upcasting" internally:
1. True upcasting `dyn SubTrait` -> `dyn SuperTrait` (this also allows
   dropping auto traits and adjusting lifetimes, as below).
2. Dropping auto traits and adjusting the lifetimes of dyn trait
   *without changing the principal[^1]*:
   `dyn Trait + AutoTraits... + 'a` -> `dyn Trait + NewAutoTraits... + 'b`
   when `AutoTraits` ⊇ `NewAutoTraits`, and `'a: 'b`.

These may seem like different operations, since (1.) includes adjusting the
vtable of a dyn trait, while (2.) is a no-op. However, to the type system,
these are handled with much the same code.

This built-in implementation of `Unsize` is the most involved, particularly
after [it was reworked](https://github.com/rust-lang/rust/pull/114036) to
support the complexities of associated types.

Specifically, the upcasting algorithm involves: For each supertrait of the
source dyn trait's principal (including itself)...
1. Unify the super trait ref with the principal of the target (making sure
   we only ever upcast to a true supertrait, and never [via an impl]).
2. For every auto trait in the source, check that it's present in the principal
   (allowing us to drop auto traits, but never gain new ones).
3. For every projection in the source, check that it unifies with a single
   projection in the target (since there may be more than one given
   `trait Sub: Sup<.., A = i32> + Sup<.., A = u32>`).

[via an impl]: https://github.com/rust-lang/rust/blob/f3457dbf84cd86d284454d12705861398ece76c3/tests/ui/traits/trait-upcasting/illegal-upcast-from-impl.rs#L19

Specifically, (3.) prevents a choice of projection bound to guide inference
unnecessarily, though it may guide inference when it is unambiguous.

[^1]: The principal is the one non-auto trait of a `dyn Trait`.