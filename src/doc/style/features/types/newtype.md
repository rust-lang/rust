% The newtype pattern

A "newtype" is a tuple or `struct` with a single field. The terminology is borrowed from Haskell.

Newtypes are a zero-cost abstraction: they introduce a new, distinct name for an
existing type, with no runtime overhead when converting between the two types.

### Use newtypes to provide static distinctions. [FIXME: needs RFC]

Newtypes can statically distinguish between different interpretations of an
underlying type.

For example, a `f64` value might be used to represent a quantity in miles or in
kilometers. Using newtypes, we can keep track of the intended interpretation:

```rust
struct Miles(pub f64);
struct Kilometers(pub f64);

impl Miles {
    fn as_kilometers(&self) -> Kilometers { ... }
}
impl Kilometers {
    fn as_miles(&self) -> Miles { ... }
}
```

Once we have separated these two types, we can statically ensure that we do not
confuse them. For example, the function

```rust
fn are_we_there_yet(distance_travelled: Miles) -> bool { ... }
```

cannot accidentally be called with a `Kilometers` value. The compiler will
remind us to perform the conversion, thus averting certain
[catastrophic bugs](http://en.wikipedia.org/wiki/Mars_Climate_Orbiter).

### Use newtypes with private fields for hiding. [FIXME: needs RFC]

A newtype can be used to hide representation details while making precise
promises to the client.

For example, consider a function `my_transform` that returns a compound iterator
type `Enumerate<Skip<vec::MoveItems<T>>>`. We wish to hide this type from the
client, so that the client's view of the return type is roughly `Iterator<(uint,
T)>`. We can do so using the newtype pattern:

```rust
struct MyTransformResult<T>(Enumerate<Skip<vec::MoveItems<T>>>);
impl<T> Iterator<(uint, T)> for MyTransformResult<T> { ... }

fn my_transform<T, Iter: Iterator<T>>(iter: Iter) -> MyTransformResult<T> {
    ...
}
```

Aside from simplifying the signature, this use of newtypes allows us to make a
expose and promise less to the client. The client does not know _how_ the result
iterator is constructed or represented, which means the representation can
change in the future without breaking client code.

> **[FIXME]** Interaction with auto-deref.

### Use newtypes to provide cost-free _views_ of another type. **[FIXME]**

> **[FIXME]** Describe the pattern of using newtypes to provide a new set of
> inherent or trait methods, providing a different perspective on the underlying
> type.
