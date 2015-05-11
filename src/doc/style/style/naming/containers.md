% Common container/wrapper methods [FIXME: needs RFC]

Containers, wrappers, and cells all provide ways to access the data
they enclose.  Accessor methods often have variants to access the data
by value, by reference, and by mutable reference.

In general, the `get` family of methods is used to access contained
data without any risk of thread failure; they return `Option` as
appropriate. This name is chosen rather than names like `find` or
`lookup` because it is appropriate for a wider range of container types.

#### Containers

For a container with keys/indexes of type `K` and elements of type `V`:

```rust
// Look up element without failing
fn get(&self, key: K) -> Option<&V>
fn get_mut(&mut self, key: K) -> Option<&mut V>

// Convenience for .get(key).map(|elt| elt.clone())
fn get_clone(&self, key: K) -> Option<V>

// Lookup element, failing if it is not found:
impl Index<K, V> for Container { ... }
impl IndexMut<K, V> for Container { ... }
```

#### Wrappers/Cells

Prefer specific conversion functions like `as_bytes` or `into_vec` whenever
possible. Otherwise, use:

```rust
// Extract contents without failing
fn get(&self) -> &V
fn get_mut(&mut self) -> &mut V
fn unwrap(self) -> V
```

#### Wrappers/Cells around `Copy` data

```rust
// Extract contents without failing
fn get(&self) -> V
```

#### `Option`-like types

Finally, we have the cases of types like `Option` and `Result`, which
play a special role for failure.

For `Option<V>`:

```rust
// Extract contents or fail if not available
fn assert(self) -> V
fn expect(self, &str) -> V
```

For `Result<V, E>`:

```rust
// Extract the contents of Ok variant; fail if Err
fn assert(self) -> V

// Extract the contents of Err variant; fail if Ok
fn assert_err(self) -> E
```
