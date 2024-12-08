Test cases intended to document behavior and try to exhaustively
explore the combinations.

## Confidence

These tests are not yet considered 100% normative, in that some
aspects of the current behavior are not desirable. This is expressed
in the "confidence" field in the following table. Values:

| Confidence | Interpretation |
| --- | --- |
| 100% | this will remain recommended behavior |
| 75% | unclear whether we will continue to accept this |
| 50% | this will likely be deprecated but remain valid |
| 25% | this could change in the future |
| 0% | this is definitely bogus and will likely change in the future in *some* way |

## Tests

| Test file | `Self` type | Pattern | Current elision behavior | Confidence |
| --- | --- | --- | --- | --- |
| `self.rs` | `Struct` | `Self` | ignore `self` parameter | 100% |
| `struct.rs` | `Struct` | `Struct` | ignore `self` parameter | 100% |
| `alias.rs` | `Struct` | `Alias` | ignore `self` parameter | 100% |
| `ref-self.rs` | `Struct` | `&Self` | take lifetime from `&Self` | 100% |
| `ref-mut-self.rs` | `Struct` | `&mut Self` | take lifetime from `&mut Self` | 100% |
| `ref-struct.rs` | `Struct` | `&Struct` | take lifetime from `&Self` | 50% |
| `ref-mut-struct.rs` | `Struct` | `&mut Struct` | take lifetime from `&mut Self` | 50% |
| `ref-alias.rs` | `Struct` | `&Alias` | ignore `Alias` | 0% |
| `ref-mut-alias.rs` | `Struct` | `&mut Alias` | ignore `Alias` | 0% |
| `lt-self.rs` | `Struct<'a>` | `Self` | ignore `Self` (and hence `'a`) | 25% |
| `lt-struct.rs` | `Struct<'a>` | `Self` | ignore `Self` (and hence `'a`) | 0% |
| `lt-alias.rs`   | `Alias<'a>` | `Self` | ignore `Self` (and hence `'a`) | 0% |
| `lt-ref-self.rs` | `Struct<'a>` | `&Self` | take lifetime from `&Self` | 75% |

In each case, we test the following patterns:

- `self: XXX`
- `self: Box<XXX>`
- `self: Pin<XXX>`
- `self: Box<Box<XXX>>`
- `self: Box<Pin<XXX>>`

In the non-reference cases, `Pin` causes errors so we substitute `Rc`.

### `async fn`

For each of the tests above we also check that `async fn` behaves as an `fn` would.
These tests are in files named `*-async.rs`.

Legends:
- ✓ ⟹ Yes / Pass
- X ⟹ No
- α ⟹ lifetime mismatch
- β ⟹ cannot infer an appropriate lifetime
- γ ⟹ missing lifetime specifier

| `async` file | Pass? | Conforms to `fn`? | How does it diverge? <br/> `fn` ⟶ `async fn` |
| --- | --- | --- | --- |
| `self-async.rs` | ✓ | ✓ | N/A |
| `struct-async.rs`| ✓ | ✓ | N/A |
| `alias-async.rs`| ✓ | ✓ | N/A |
| `assoc-async.rs`| ✓ | ✓ | N/A |
| `ref-self-async.rs` | X | ✓ | N/A |
| `ref-mut-self-async.rs` | X | ✓ | N/A |
| `ref-struct-async.rs` | X | ✓ | N/A |
| `ref-mut-struct-async.rs` | X | ✓ | N/A |
| `ref-alias-async.rs` | ✓ | ✓ | N/A |
| `ref-assoc-async.rs` | ✓ | ✓ | N/A |
| `ref-mut-alias-async.rs` | ✓ | ✓ | N/A |
| `lt-self-async.rs` | ✓ | ✓ | N/A
| `lt-struct-async.rs` | ✓ | ✓ | N/A
| `lt-alias-async.rs` | ✓ | ✓ | N/A
| `lt-assoc-async.rs` | ✓ | ✓ | N/A
| `lt-ref-self-async.rs` | X | ✓ | N/A |
