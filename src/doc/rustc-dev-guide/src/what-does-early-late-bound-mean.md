# Early and Late Bound Parameter Definitions

Understanding this page likely requires a rudimentary understanding of higher ranked
trait bounds/`for<'a>`and also what types such as `dyn for<'a> Trait<'a>` and
 `for<'a> fn(&'a u32)` mean. Reading [the nomincon chapter](https://doc.rust-lang.org/nomicon/hrtb.html)
on HRTB may be useful for understanding this syntax. The meaning of `for<'a> fn(&'a u32)`
is incredibly similar to the meaning of `T: for<'a> Trait<'a>`.

If you are looking for information on the `RegionKind` variants `ReLateBound` and `ReEarlyBound`
you should look at the section on [bound vars and params](./bound-vars-and-params.md). This section
discusses what makes generic parameters on functions and closures late/early bound. Not the general
concept of bound vars and generic parameters which `RegionKind` has named somewhat confusingly
with this topic.

## What does it mean for parameters to be early or late bound

All function definitions conceptually have a ZST (this is represented by `TyKind::FnDef` in rustc).
The only generics on this ZST are the early bound parameters of the function definition. e.g.
```rust
fn foo<'a>(_: &'a u32) {}

fn main() {
    let b = foo;
    //  ^ `b` has type `FnDef(foo, [])` (no args because `'a` is late bound)
    assert!(std::mem::size_of_val(&b) == 0);
}
```

In order to call `b` the late bound parameters do need to be provided, these are inferred at the
call site instead of when we refer to `foo`.
```rust
fn main() {
    let b = foo;
    let a: &'static u32 = &10;
    foo(a);
    // the lifetime argument for `'a` on `foo` is inferred at the callsite
    // the generic parameter `'a` on `foo` is inferred to `'static` here
}
```

Because late bound parameters are not part of the `FnDef`'s args this allows us to prove trait
bounds such as `F: for<'a> Fn(&'a u32)` where `F` is `foo`'s `FnDef`. e.g.
```rust
fn foo_early<'a, T: Trait<'a>>(_: &'a u32, _: T) {}
fn foo_late<'a, T>(_: &'a u32, _: T) {}

fn accepts_hr_func<F: for<'a> Fn(&'a u32, u32)>(_: F) {}

fn main() {
    // doesnt work, the substituted bound is `for<'a> FnDef<'?0>: Fn(&'a u32, u32)`
    // `foo_early` only implements `for<'a> FnDef<'a>: Fn(&'a u32, u32)`- the lifetime
    // of the borrow in the function argument must be the same as the lifetime
    // on the `FnDef`.
    accepts_hr_func(foo_early);

    // works, the substituted bound is `for<'a> FnDef: Fn(&'a u32, u32)`
    accepts_hr_func(foo_late);
}

// the builtin `Fn` impls for `foo_early` and `foo_late` look something like:
// `foo_early`
impl<'a, T: Trait<'a>> Fn(&'a u32, T) for FooEarlyFnDef<'a, T> { ... }
// `foo_late`
impl<'a, T> Fn(&'a u32, T) for FooLateFnDef<T> { ... }

```

Early bound parameters are present on the `FnDef`. Late bound generic parameters are not present
on the `FnDef` but are instead constrained by the builtin `Fn*` impl.

The same distinction applies to closures. Instead of `FnDef` we are talking about the anonymous
closure type. Closures are [currently unsound](https://github.com/rust-lang/rust/issues/84366) in
ways that are closely related to the distinction between early/late bound
parameters (more on this later)

The early/late boundness of generic parameters is only relevent for the desugaring of
functions/closures into types with builtin `Fn*` impls. It does not make sense to talk about
in other contexts.

The `generics_of` query in rustc only contains early bound parameters. In this way it acts more
like `generics_of(my_func)` is the generics for the FnDef than the generics provided to the function
body although it's not clear to the author of this section if this was the actual justification for
making `generics_of` behave this way.

## What parameters are currently late bound

Below are the current requirements for determining if a generic parameter is late bound. It is worth
keeping in mind that these are not necessarily set in stone and it is almost certainly possible to
be more flexible.

### Must be a lifetime parameter

Rust can't support types such as `for<T> dyn Trait<T>` or `for<T> fn(T)`, this is a
fundamental limitation of the language as we are required to monomorphize type/const
parameters and cannot do so behind dynamic dispatch. (technically we could probably
support `for<T> dyn MarkerTrait<T>` as there is nothing to monomorphize)

Not being able to support `for<T> dyn Trait<T>` resulted in making all type and const
parameters early bound. Only lifetime parameters can be late bound.

### Must not appear in the where clauses

In order for a generic parameter to be late bound it must not appear in any where clauses.
This is currently an incredibly simplistic check that causes lifetimes to be early bound even
if the where clause they appear in are always true, or implied by well formedness of function
arguments. e.g.
```rust
fn foo1<'a: 'a>(_: &'a u32) {}
//     ^^ early bound parameter because it's in a `'a: 'a` clause
//        even though the bound obviously holds all the time
fn foo2<'a, T: Trait<'a>(a: T, b: &'a u32) {}
//     ^^ early bound parameter because it's used in the `T: Trait<'a>` clause
fn foo3<'a, T: 'a>(_: &'a T) {}
//     ^^ early bound parameter because it's used in the `T: 'a` clause
//        even though that bound is implied by wellformedness of `&'a T`
fn foo4<'a, 'b: 'a>(_: Inv<&'a ()>, _: Inv<&'b ()>) {}
//      ^^  ^^         ^^^ note:
//      ^^  ^^         `Inv` stands for `Invariant` and is used to
//      ^^  ^^          make the the type parameter invariant. This
//      ^^  ^^          is necessary for demonstration purposes as
//      ^^  ^^          `for<'a, 'b> fn(&'a (), &'b ())` and
//      ^^  ^^          `for<'a> fn(&'a u32, &'a u32)` are subtypes-
//      ^^  ^^          of eachother which makes the bound trivially
//      ^^  ^^          satisfiable when making the fnptr. `Inv`
//      ^^  ^^          disables this subtyping.
//      ^^  ^^
//      ^^^^^^ both early bound parameters because they are present in the
//            `'b: 'a` clause
```

The reason for this requirement is that we cannot represent the `T: Trait<'a>` or `'a: 'b` clauses
on a function pointer. `for<'a, 'b> fn(Inv<&'a ()>, Inv<&'b ()>)` is not a valid function pointer to
represent`foo4` as it would allow calling the function without `'b: 'a` holding.

### Must be constrained by where clauses or function argument types

The builtin impls of the `Fn*` traits for closures and `FnDef`s cannot not have any unconstrained
parameters. For example the following impl is illegal:
```rust
impl<'a> Trait for u32 { type Assoc = &'a u32; }
```
We must not end up with a similar impl for the `Fn*` traits e.g.
```rust
impl<'a> Fn<()> for FnDef { type Assoc = &'a u32 }
```

Violating this rule can trivially lead to unsoundness as seen in [#84366](https://github.com/rust-lang/rust/issues/84366).
Additionally if we ever support late bound type params then an impl like:
```rust
impl<T> Fn<()> for FnDef { type Assoc = T; }
```
would break the compiler in various ways.

In order to ensure that everything functions correctly, we do not allow generic parameters to
be late bound if it would result in a builtin impl that does not constrain all of the generic
parameters on the builtin impl. Making a generic parameter be early bound trivially makes it be
constrained by the builtin impl as it ends up on the self type.

Because of the requirement that late bound parameters must not appear in where clauses, checking
this is simpler than the rules for checking impl headers constrain all the parameters on the impl.
We only have to ensure that all late bound parameters appear at least once in the function argument
types outside of an alias (e.g. an associated type).

The requirement that they not indirectly be in the args of an alias for it to count is the
same as why the follow code is forbidden:
```rust
impl<T: Trait> OtherTrait for <T as Trait>::Assoc { type Assoc = T }
```
There is no guarantee that `<T as Trait>::Assoc` will normalize to different types for every
instantiation of `T`. If we were to allow this impl we could get overlapping impls and the
same is true of the builtin `Fn*` impls.

## Making more generic parameters late bound

It is generally considered desirable for more parameters to be late bound as it makes
the builtin `Fn*` impls more flexible. Right now many of the requirements for making
a parameter late bound are overly restrictive as they are tied to what we can currently
(or can ever) do with fn ptrs.

It would be theoretically possible to support late bound params in `where`-clauses in the
language by introducing implication types which would allow us to express types such as:
`for<'a, 'b: 'a> fn(Inv<&'a u32>, Inv<&'b u32>)` which would ensure `'b: 'a` is upheld when
calling the function pointer.

It would also be theoretically possible to support it by making the coercion to a fn ptr
instantiate the parameter with an infer var while still allowing the FnDef to not have the
generic parameter present as trait impls are perfectly capable of representing the where clauses
on the function on the impl itself. This would also allow us to support late bound type/const
vars allowing bounds like `F: for<T> Fn(T)` to hold.

It is almost somewhat unclear if we can change the `Fn` traits to be structured differently
so that we never have to make a parameter early bound just to make the builtin impl have all
generics be constrained. Of all the possible causes of a generic parameter being early bound
this seems the most difficult to remove.

Whether these would be good ideas to implement is a separate question- they are only brought
up to illustrate that the current rules are not necessarily set in stone and a result of
"its the only way of doing this".

