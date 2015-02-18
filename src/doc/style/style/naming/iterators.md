% Iterators

#### Method names [RFC #199]

> The guidelines below were approved by [RFC #199](https://github.com/rust-lang/rfcs/pull/199).

For a container with elements of type `U`, iterator methods should be named:

```rust
fn iter(&self) -> T           // where T implements Iterator<&U>
fn iter_mut(&mut self) -> T   // where T implements Iterator<&mut U>
fn into_iter(self) -> T       // where T implements Iterator<U>
```

The default iterator variant yields shared references `&U`.

#### Type names [RFC #344]

> The guidelines below were approved by [RFC #344](https://github.com/rust-lang/rfcs/pull/344).

The name of an iterator type should be the same as the method that
produces the iterator.

For example:

* `iter` should yield an `Iter`
* `iter_mut` should yield an `IterMut`
* `into_iter` should yield an `IntoIter`
* `keys` should yield `Keys`

These type names make the most sense when prefixed with their owning module,
e.g. `vec::IntoIter`.
