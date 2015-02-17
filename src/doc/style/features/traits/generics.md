% Using traits for bounds on generics

The most widespread use of traits is for writing generic functions or types. For
example, the following signature describes a function for consuming any iterator
yielding items of type `A` to produce a collection of `A`:

```rust
fn from_iter<T: Iterator<A>>(iterator: T) -> SomeCollection<A>
```

Here, the `Iterator` trait is specifies an interface that a type `T` must
explicitly implement to be used by this generic function.

**Pros**:

* _Reusability_. Generic functions can be applied to an open-ended collection of
  types, while giving a clear contract for the functionality those types must
  provide.
* _Static dispatch and optimization_. Each use of a generic function is
  specialized ("monomorphized") to the particular types implementing the trait
  bounds, which means that (1) invocations of trait methods are static, direct
  calls to the implementation and (2) the compiler can inline and otherwise
  optimize these calls.
* _Inline layout_. If a `struct` and `enum` type is generic over some type
  parameter `T`, values of type `T` will be laid out _inline_ in the
  `struct`/`enum`, without any indirection.
* _Inference_. Since the type parameters to generic functions can usually be
  inferred, generic functions can help cut down on verbosity in code where
  explicit conversions or other method calls would usually be necessary. See the
  [overloading/implicits use case](#use-case:-limited-overloading-and/or-implicit-conversions)
  below.
* _Precise types_. Because generic give a _name_ to the specific type
  implementing a trait, it is possible to be precise about places where that
  exact type is required or produced. For example, a function

  ```rust
  fn binary<T: Trait>(x: T, y: T) -> T
  ```

  is guaranteed to consume and produce elements of exactly the same type `T`; it
  cannot be invoked with parameters of different types that both implement
  `Trait`.

**Cons**:

* _Code size_. Specializing generic functions means that the function body is
  duplicated. The increase in code size must be weighed against the performance
  benefits of static dispatch.
* _Homogeneous types_. This is the other side of the "precise types" coin: if
  `T` is a type parameter, it stands for a _single_ actual type. So for example
  a `Vec<T>` contains elements of a single concrete type (and, indeed, the
  vector representation is specialized to lay these out in line). Sometimes
  heterogeneous collections are useful; see
  [trait objects](#use-case:-trait-objects) below.
* _Signature verbosity_. Heavy use of generics can bloat function signatures.
  **[Ed. note]** This problem may be mitigated by some language improvements; stay tuned.

### Favor widespread traits. **[FIXME: needs RFC]**

Generic types are a form of abstraction, which entails a mental indirection: if
a function takes an argument of type `T` bounded by `Trait`, clients must first
think about the concrete types that implement `Trait` to understand how and when
the function is callable.

To keep the cost of abstraction low, favor widely-known traits. Whenever
possible, implement and use traits provided as part of the standard library.  Do
not introduce new traits for generics lightly; wait until there are a wide range
of types that can implement the type.
