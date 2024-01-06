# Rustdoc search

Rustdoc Search is two programs: `search_index.rs`
and `search.js`. The first generates a nasty JSON
file with a full list of items and function signatures
in the crates in the doc bundle, and the second reads
it, turns it into some in-memory structures, and
scans them linearly to search.

<!-- toc -->

## Search index format

`search.js` calls this Raw, because it turns it into
a more normal object tree after loading it.
Naturally, it's also written without newlines or spaces.

```json
[
    [ "crate_name", {
        "doc": "Documentation",
        "n": ["function_name", "Data"],
        "t": "HF",
        "d": ["This function gets the name of an integer with Data", "The data struct"],
        "q": [[0, "crate_name"]],
        "i": [2, 0],
        "p": [[1, "i32"], [1, "str"], [5, "crate_name::Data"]],
        "f": "{{gb}{d}}`",
        "b": [],
        "c": [],
        "a": [["get_name", 0]],
    }]
]
```

[`src/librustdoc/html/static/js/externs.js`]
defines an actual schema in a Closure `@typedef`.

The above index defines a crate called `crate_name`
with a free function called `function_name` and a struct called `Data`,
with the type signature `Data, i32 -> str`,
and an alias, `get_name`, that equivalently refers to `function_name`.

[`src/librustdoc/html/static/js/externs.js`]: https://github.com/rust-lang/rust/blob/79b710c13968a1a48d94431d024d2b1677940866/src/librustdoc/html/static/js/externs.js#L204-L258

The search index needs to fit the needs of the `rustdoc` compiler,
the `search.js` frontend,
and also be compact and fast to decode.
It makes a lot of compromises:

* The `rustdoc` compiler runs on one crate at a time,
  so each crate has an essentially separate search index.
  It [merges] them by having each crate on one line
  and looking at the first quoted string.
* Names in the search index are given
  in their original case and with underscores.
  When the search index is loaded,
  `search.js` stores the original names for display,
  but also folds them to lowercase and strips underscores for search.
  You'll see them called `normalized`.
* The `f` array stores types as offsets into the `p` array.
  These types might actually be from another crate,
  so `search.js` has to turn the numbers into names and then
  back into numbers to deduplicate them if multiple crates in the
  same index mention the same types.
* It's a JSON file, but not designed to be human-readable.
  Browsers already include an optimized JSON decoder,
  so this saves on `search.js` code and performs better for small crates,
  but instead of using objects like normal JSON formats do,
  it tries to put data of the same type next to each other
  so that the sliding window used by [DEFLATE] can find redundancies.
  Where `search.js` does its own compression,
  it's designed to save memory when the file is finally loaded,
  not just size on disk or network transfer.

[merges]: https://github.com/rust-lang/rust/blob/79b710c13968a1a48d94431d024d2b1677940866/src/librustdoc/html/render/write_shared.rs#L151-L164
[DEFLATE]: https://en.wikipedia.org/wiki/Deflate

### Parallel arrays and indexed maps

Most data in the index
(other than `doc`, which is a single string for the whole crate,
`p`, which is a separate structure
and `a`, which is also a separate structure)
is a set of parallel arrays defining each searchable item.

For example,
the above search index can be turned into this table:

| n | t | d | q | i | f | b | c |
|---|---|---|---|---|---|---|---|
| `function_name` | `H` | This function gets the name of an integer with Data | `crate_name` | 2 | `{{gb}{d}}` | NULL | NULL |
| `Data` | `F` | The data struct | `crate_name` | 0 | `` ` `` | NULL | NULL |

The above code doesn't use `c`, which holds deprecated indices,
or `b`, which maps indices to strings.
If `crate_name::function_name` used both, it would look like this.

```json
        "b": [[0, "impl-Foo-for-Bar"]],
        "c": [0],
```

This attaches a disambiguator to index 0 and marks it deprecated.

The advantage of this layout is that these APIs often have implicit structure
that DEFLATE can take advantage of,
but that rustdoc can't assume.
Like how names are usually CamelCase or snake_case,
but descriptions aren't.

`q` is a Map from *the first applicable* ID to a parent module path.
This is a weird trick, but it makes more sense in pseudo-code:

```rust
let mut parent_module = "";
for (i, entry) in search_index.iter().enumerate() {
    if q.contains(i) {
        parent_module = q.get(i);
    }
    // ... do other stuff with `entry` ...
}
```

This is valid because everything has a parent module
(even if it's just the crate itself),
and is easy to assemble because the rustdoc generator sorts by path
before serializing.
Doing this allows rustdoc to not only make the search index smaller,
but reuse the same string representing the parent path across multiple in-memory items.

### `i`, `f`, and `p`

`i` and `f` both index into `p`, the array of parent items.

`i` is just a one-indexed number
(not zero-indexed because `0` is used for items that have no parent item).
It's different from `q` because `q` represents the parent *module or crate*,
which everything has,
while `i`/`q` are used for *type and trait-associated items* like methods.

`f`, the function signatures, use their own encoding.

```ebnf
f = { FItem | FBackref }
FItem = FNumber | ( '{', {FItem}, '}' )
FNumber = { '@' | 'A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'G' | 'H' | 'I' | 'J' | 'K' | 'L' | 'M' | 'N' | 'O' }, ( '`' | 'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'g' | 'h' | 'i' | 'j' | 'k ' | 'l' | 'm' | 'n' | 'o' )
FBackref = ( '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | ':' | ';' | '<' | '=' | '>' | '?' )
```

An FNumber is a variable-length, self-terminating base16 number
(terminated because the last hexit is lowercase while all others are uppercase).
These are one-indexed references into `p`, because zero is used for nulls,
and negative numbers represent generics.
The sign bit is represented using [zig-zag encoding]
(the internal object representation also uses negative numbers,
even after decoding,
to represent generics).
This alphabet is chosen because the characters can be turned into hexits by
masking off the last four bits of the ASCII encoding.

For example, `{{gb}{d}}` is equivalent to the json `[[3, 1], [2]]`.
Because of zigzag encoding, `` ` `` is +0, `a` is -0 (which is not used),
`b` is +1, and `c` is -1.

[empirically]: https://github.com/rust-lang/rust/pull/83003
[zig-zag encoding]: https://en.wikipedia.org/wiki/Variable-length_quantity#Zigzag_encoding

## Searching by name

Searching by name works by looping through the search index
and running these functions on each:

* [`editDistance`] is always used to determine a match
  (unless quotes are specified, which would use simple equality instead).
  It computes the number of swaps, inserts, and removes needed to turn
  the query name into the entry name.
  For example, `foo` has zero distance from itself,
  but a distance of 1 from `ofo` (one swap) and `foob` (one insert).
  It is checked against an heuristic threshold, and then,
  if it is within that threshold, the distance is stored for ranking.
* [`String.prototype.indexOf`] is always used to determine a match.
  If it returns anything other than -1, the result is added,
  even if `editDistance` exceeds its threshold,
  and the index is stored for ranking.
* [`checkPath`] is used if, and only if, a parent path is specified
  in the query. For example, `vec` has no parent path, but `vec::vec` does.
  Within checkPath, editDistance and indexOf are used,
  and the path query has its own heuristic threshold, too.
  If it's not within the threshold, the entry is rejected,
  even if the first two pass.
  If it's within the threshold, the path distance is stored
  for ranking.
* [`checkType`] is used only if there's a type filter,
  like the struct in `struct:vec`. If it fails,
  the entry is rejected.

If all four criteria pass
(plus the crate filter, which isn't technically part of the query),
the results are sorted by [`sortResults`].

[`editDistance`]: https://github.com/rust-lang/rust/blob/79b710c13968a1a48d94431d024d2b1677940866/src/librustdoc/html/static/js/search.js#L137
[`String.prototype.indexOf`]: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/indexOf
[`checkPath`]: https://github.com/rust-lang/rust/blob/79b710c13968a1a48d94431d024d2b1677940866/src/librustdoc/html/static/js/search.js#L1814
[`checkType`]: https://github.com/rust-lang/rust/blob/79b710c13968a1a48d94431d024d2b1677940866/src/librustdoc/html/static/js/search.js#L1787
[`sortResults`]: https://github.com/rust-lang/rust/blob/79b710c13968a1a48d94431d024d2b1677940866/src/librustdoc/html/static/js/search.js#L1229

## Searching by type

Searching by type can be divided into two phases,
and the second phase has two sub-phases.

* Turn names in the query into numbers.
* Loop over each entry in the search index:
   * Quick rejection using a bloom filter.
   * Slow rejection using a recursive type unification algorithm.

In the names->numbers phase, if the query has only one name in it,
the editDistance function is used to find a near match if the exact match fails,
but if there's multiple items in the query,
non-matching items are treated as generics instead.
This means `hahsmap` will match hashmap on its own, but `hahsmap, u32`
is going to match the same things `T, u32` matches
(though rustdoc will detect this particular problem and warn about it).

Then, when actually looping over each item,
the bloom filter will probably reject entries that don't have every
type mentioned in the query.
For example, the bloom query allows a query of `i32 -> u32` to match
a function with the type `i32, u32 -> bool`,
but unification will reject it later.

The unification filter ensures that:

* Bag semantics are respected. If you query says `i32, i32`,
  then the function has to mention *two* i32s, not just one.
* Nesting semantics are respected. If your query says `vec<option>`,
  then `vec<option<i32>>` is fine, but `option<vec<i32>>` *is not* a match.
* The division between return type and parameter is respected.
  `i32 -> u32` and `u32 -> i32` are completely different.

The bloom filter checks none of these things,
and, on top of that, can have false positives.
But it's fast and uses very little memory, so the bloom filter helps.
