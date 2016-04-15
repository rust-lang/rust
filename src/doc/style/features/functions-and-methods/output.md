% Output from functions and methods

### Don't overpromise. [FIXME]

> **[FIXME]** Add discussion of overly-specific return types,
> e.g. returning a compound iterator type rather than hiding it behind
> a use of newtype.

### Let clients choose what to throw away. [FIXME: needs RFC]

#### Return useful intermediate results:

Many functions that answer a question also compute interesting related data.  If
this data is potentially of interest to the client, consider exposing it in the
API.

Prefer

```rust,ignore
struct SearchResult {
    found: bool,          // item in container?
    expected_index: usize // what would the item's index be?
}

fn binary_search(&self, k: Key) -> SearchResult
```
or

```rust,ignore
fn binary_search(&self, k: Key) -> (bool, usize)
```

over

```rust,ignore
fn binary_search(&self, k: Key) -> bool
```

#### Yield back ownership:

Prefer

```rust,ignore
fn from_utf8_owned(vv: Vec<u8>) -> Result<String, Vec<u8>>
```

over

```rust,ignore
fn from_utf8_owned(vv: Vec<u8>) -> Option<String>
```

The `from_utf8_owned` function gains ownership of a vector.  In the successful
case, the function consumes its input, returning an owned string without
allocating or copying. In the unsuccessful case, however, the function returns
back ownership of the original slice.
