# Variance of type and lifetime parameters

For a more general background on variance, see the [background] appendix.

[background]: ./appendix/background.html

During type checking we must infer the variance of type and lifetime
parameters. The algorithm is taken from Section 4 of the paper ["Taming the
Wildcards: Combining Definition- and Use-Site Variance"][pldi11] published in
PLDI'11 and written by Altidor et al., and hereafter referred to as The Paper.

[pldi11]: https://people.cs.umass.edu/~yannis/variance-extended2011.pdf

This inference is explicitly designed *not* to consider the uses of
types within code. To determine the variance of type parameters
defined on type `X`, we only consider the definition of the type `X`
and the definitions of any types it references.

We only infer variance for type parameters found on *data types*
like structs and enums. In these cases, there is a fairly straightforward
explanation for what variance means. The variance of the type
or lifetime parameters defines whether `T<A>` is a subtype of `T<B>`
(resp. `T<'a>` and `T<'b>`) based on the relationship of `A` and `B`
(resp. `'a` and `'b`).

We do not infer variance for type parameters found on traits, functions,
or impls. Variance on trait parameters can indeed make sense
(and we used to compute it) but it is actually rather subtle in
meaning and not that useful in practice, so we removed it. See the
[addendum] for some details. Variances on function/impl parameters, on the
other hand, doesn't make sense because these parameters are instantiated and
then forgotten, they don't persist in types or compiled byproducts.

[addendum]: #addendum

> **Notation**
>
> We use the notation of The Paper throughout this chapter:
>
> - `+` is _covariance_.
> - `-` is _contravariance_.
> - `*` is _bivariance_.
> - `o` is _invariance_.

## The algorithm

The basic idea is quite straightforward. We iterate over the types
defined and, for each use of a type parameter `X`, accumulate a
constraint indicating that the variance of `X` must be valid for the
variance of that use site. We then iteratively refine the variance of
`X` until all constraints are met. There is *always* a solution, because at
the limit we can declare all type parameters to be invariant and all
constraints will be satisfied.

As a simple example, consider:

```rust,ignore
enum Option<A> { Some(A), None }
enum OptionalFn<B> { Some(|B|), None }
enum OptionalMap<C> { Some(|C| -> C), None }
```

Here, we will generate the constraints:

```text
1. V(A) <= +
2. V(B) <= -
3. V(C) <= +
4. V(C) <= -
```

These indicate that (1) the variance of A must be at most covariant;
(2) the variance of B must be at most contravariant; and (3, 4) the
variance of C must be at most covariant *and* contravariant. All of these
results are based on a variance lattice defined as follows:

```text
   *      Top (bivariant)
-     +
   o      Bottom (invariant)
```

Based on this lattice, the solution `V(A)=+`, `V(B)=-`, `V(C)=o` is the
optimal solution. Note that there is always a naive solution which
just declares all variables to be invariant.

You may be wondering why fixed-point iteration is required. The reason
is that the variance of a use site may itself be a function of the
variance of other type parameters. In full generality, our constraints
take the form:

```text
V(X) <= Term
Term := + | - | * | o | V(X) | Term x Term
```

Here the notation `V(X)` indicates the variance of a type/region
parameter `X` with respect to its defining class. `Term x Term`
represents the "variance transform" as defined in the paper:

>  If the variance of a type variable `X` in type expression `E` is `V2`
  and the definition-site variance of the corresponding type parameter
  of a class `C` is `V1`, then the variance of `X` in the type expression
  `C<E>` is `V3 = V1.xform(V2)`.

## Constraints

If I have a struct or enum with where clauses:

```rust,ignore
struct Foo<T: Bar> { ... }
```

you might wonder whether the variance of `T` with respect to `Bar` affects the
variance `T` with respect to `Foo`. I claim no.  The reason: assume that `T` is
invariant with respect to `Bar` but covariant with respect to `Foo`. And then
we have a `Foo<X>` that is upcast to `Foo<Y>`, where `X <: Y`. However, while
`X : Bar`, `Y : Bar` does not hold.  In that case, the upcast will be illegal,
but not because of a variance failure, but rather because the target type
`Foo<Y>` is itself just not well-formed. Basically we get to assume
well-formedness of all types involved before considering variance.

### Dependency graph management

Because variance is a whole-crate inference, its dependency graph
can become quite muddled if we are not careful. To resolve this, we refactor
into two queries:

- `crate_variances` computes the variance for all items in the current crate.
- `variances_of` accesses the variance for an individual reading; it
  works by requesting `crate_variances` and extracting the relevant data.

If you limit yourself to reading `variances_of`, your code will only
depend then on the inference of that particular item.

Ultimately, this setup relies on the [red-green algorithm][rga]. In particular,
every variance query effectively depends on all type definitions in the entire
crate (through `crate_variances`), but since most changes will not result in a
change to the actual results from variance inference, the `variances_of` query
will wind up being considered green after it is re-evaluated.

[rga]: ./queries/incremental-compilation.html

<a id="addendum"></a>

## Addendum: Variance on traits

As mentioned above, we used to permit variance on traits. This was
computed based on the appearance of trait type parameters in
method signatures and was used to represent the compatibility of
vtables in trait objects (and also "virtual" vtables or dictionary
in trait bounds). One complication was that variance for
associated types is less obvious, since they can be projected out
and put to myriad uses, so it's not clear when it is safe to allow
`X<A>::Bar` to vary (or indeed just what that means). Moreover (as
covered below) all inputs on any trait with an associated type had
to be invariant, limiting the applicability. Finally, the
annotations (`MarkerTrait`, `PhantomFn`) needed to ensure that all
trait type parameters had a variance were confusing and annoying
for little benefit.

Just for historical reference, I am going to preserve some text indicating how
one could interpret variance and trait matching.

### Variance and object types

Just as with structs and enums, we can decide the subtyping
relationship between two object types `&Trait<A>` and `&Trait<B>`
based on the relationship of `A` and `B`. Note that for object
types we ignore the `Self` type parameter – it is unknown, and
the nature of dynamic dispatch ensures that we will always call a
function that is expected the appropriate `Self` type. However, we
must be careful with the other type parameters, or else we could
end up calling a function that is expecting one type but provided
another.

To see what I mean, consider a trait like so:

```rust
trait ConvertTo<A> {
    fn convertTo(&self) -> A;
}
```

Intuitively, If we had one object `O=&ConvertTo<Object>` and another
`S=&ConvertTo<String>`, then `S <: O` because `String <: Object`
(presuming Java-like "string" and "object" types, my go to examples
for subtyping). The actual algorithm would be to compare the
(explicit) type parameters pairwise respecting their variance: here,
the type parameter A is covariant (it appears only in a return
position), and hence we require that `String <: Object`.

You'll note though that we did not consider the binding for the
(implicit) `Self` type parameter: in fact, it is unknown, so that's
good. The reason we can ignore that parameter is precisely because we
don't need to know its value until a call occurs, and at that time (as
you said) the dynamic nature of virtual dispatch means the code we run
will be correct for whatever value `Self` happens to be bound to for
the particular object whose method we called. `Self` is thus different
from `A`, because the caller requires that `A` be known in order to
know the return type of the method `convertTo()`. (As an aside, we
have rules preventing methods where `Self` appears outside of the
receiver position from being called via an object.)

### Trait variance and vtable resolution

But traits aren't only used with objects. They're also used when
deciding whether a given impl satisfies a given trait bound. To set the
scene here, imagine I had a function:

```rust,ignore
fn convertAll<A,T:ConvertTo<A>>(v: &[T]) { ... }
```

Now imagine that I have an implementation of `ConvertTo` for `Object`:

```rust,ignore
impl ConvertTo<i32> for Object { ... }
```

And I want to call `convertAll` on an array of strings. Suppose
further that for whatever reason I specifically supply the value of
`String` for the type parameter `T`:

```rust,ignore
let mut vector = vec!["string", ...];
convertAll::<i32, String>(vector);
```

Is this legal? To put another way, can we apply the `impl` for
`Object` to the type `String`? The answer is yes, but to see why
we have to expand out what will happen:

- `convertAll` will create a pointer to one of the entries in the
  vector, which will have type `&String`
- It will then call the impl of `convertTo()` that is intended
  for use with objects. This has the type `fn(self: &Object) -> i32`.

  It is OK to provide a value for `self` of type `&String` because
  `&String <: &Object`.

OK, so intuitively we want this to be legal, so let's bring this back
to variance and see whether we are computing the correct result. We
must first figure out how to phrase the question "is an impl for
`Object,i32` usable where an impl for `String,i32` is expected?"

Maybe it's helpful to think of a dictionary-passing implementation of
type classes. In that case, `convertAll()` takes an implicit parameter
representing the impl. In short, we *have* an impl of type:

```text
V_O = ConvertTo<i32> for Object
```

and the function prototype expects an impl of type:

```text
V_S = ConvertTo<i32> for String
```

As with any argument, this is legal if the type of the value given
(`V_O`) is a subtype of the type expected (`V_S`). So is `V_O <: V_S`?
The answer will depend on the variance of the various parameters. In
this case, because the `Self` parameter is contravariant and `A` is
covariant, it means that:

```text
V_O <: V_S iff
    i32 <: i32
    String <: Object
```

These conditions are satisfied and so we are happy.

### Variance and associated types

Traits with associated types – or at minimum projection
expressions – must be invariant with respect to all of their
inputs. To see why this makes sense, consider what subtyping for a
trait reference means:

```text
<T as Trait> <: <U as Trait>
```

means that if I know that `T as Trait`, I also know that `U as
Trait`. Moreover, if you think of it as dictionary passing style,
it means that a dictionary for `<T as Trait>` is safe to use where
a dictionary for `<U as Trait>` is expected.

The problem is that when you can project types out from `<T as
Trait>`, the relationship to types projected out of `<U as Trait>`
is completely unknown unless `T==U` (see #21726 for more
details). Making `Trait` invariant ensures that this is true.

Another related reason is that if we didn't make traits with
associated types invariant, then projection is no longer a
function with a single result. Consider:

```rust,ignore
trait Identity { type Out; fn foo(&self); }
impl<T> Identity for T { type Out = T; ... }
```

Now if I have `<&'static () as Identity>::Out`, this can be
validly derived as `&'a ()` for any `'a`:

```text
<&'a () as Identity> <: <&'static () as Identity>
if &'static () < : &'a ()   -- Identity is contravariant in Self
if 'static : 'a             -- Subtyping rules for relations
```

This change otoh means that `<'static () as Identity>::Out` is
always `&'static ()` (which might then be upcast to `'a ()`,
separately). This was helpful in solving #21750.
