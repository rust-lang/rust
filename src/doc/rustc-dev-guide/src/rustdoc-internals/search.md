# Rustdoc search

Rustdoc Search is two programs: `search_index.rs`
and `search.js`. The first generates a nasty JSON
file with a full list of items and function signatures
in the crates in the doc bundle, and the second reads
it, turns it into some in-memory structures, and
scans them linearly to search.

## Search index format

`search.js` calls this Raw, because it turns it into
a more normal object tree after loading it.
For space savings, it's also written without newlines or spaces.

```json
[
    [ "crate_name", {
        // name
        "n": ["function_name", "Data"],
        // type
        "t": "HF",
        // parent module
        "q": [[0, "crate_name"]],
        // parent type
        "i": [2, 0],
        // type dictionary
        "p": [[1, "i32"], [1, "str"], [5, "Data", 0]],
        // function signature
        "f": "{{gb}{d}}`", // [[3, 1], [2]]
        // impl disambiguator
        "b": [],
        // deprecated flag
        "c": "OjAAAAAAAAA=", // empty bitmap
        // empty description flag
        "e": "OjAAAAAAAAA=", // empty bitmap
        // aliases
        "a": [["get_name", 0]],
        // description shards
        "D": "g", // 3
        // inlined re-exports
        "r": [],
    }]
]
```

[`src/librustdoc/html/static/js/rustdoc.d.ts`]
defines an actual schema in a TypeScript `type`.

| Key | Name                 | Description  |
| --- | -------------------- | ------------ |
| `n` | Names                | Item names   |
| `t` | Item Type            | One-char item type code |
| `q` | Parent module        | `Map<index, path>` |
| `i` | Parent type          | list of indexes |
| `f` | Function signature   | [encoded](#i-f-and-p) |
| `b` | Impl disambiguator   | `Map<index, string>` |
| `c` | Deprecation flag     | [roaring bitmap](#roaring-bitmaps) |
| `e` | Description is empty | [roaring bitmap](#roaring-bitmaps) |
| `p` | Type dictionary      | `[[item type, path]]` |
| `a` | Alias                | `Map<string, index>` |
| `D` | description shards   | [encoded](#how-descriptions-are-stored) |

The above index defines a crate called `crate_name`
with a free function called `function_name` and a struct called `Data`,
with the type signature `Data, i32 -> str`,
and an alias, `get_name`, that equivalently refers to `function_name`.

[`src/librustdoc/html/static/js/rustdoc.d.ts`]: https://github.com/rust-lang/rust/blob/2f92f050e83bf3312ce4ba73c31fe843ad3cbc60/src/librustdoc/html/static/js/rustdoc.d.ts#L344-L390

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

Abstractly, Rustdoc Search data is a table, stored in column-major form.
Most data in the index represents a set of parallel arrays
(the "columns") which refer to the same data if they're at the same position.

For example,
the above search index can be turned into this table:

|   | n | t | [d] | q | i | f | b | c |
|---|---|---|-----|---|---|---|---|---|
| 0 | `crate_name`    | `D` | Documentation | NULL | 0 | NULL | NULL | 0 |
| 1 | `function_name` | `H` | This function gets the name of an integer with Data | `crate_name` | 2 | `{{gb}{d}}` | NULL | 0 |
| 2 | `Data` | `F` | The data struct | `crate_name` | 0 | `` ` `` | NULL | 0 |

[d]: #how-descriptions-are-stored

The crate row is implied in most columns, since its type is known (it's a crate),
it can't have a parent (crates form the root of the module tree),
its name is specified as the map key,
and function-specific data like the impl disambiguator can't apply either.
However, it can still have a description and it can still be deprecated.
The crate, therefore, has a primary key of `0`.

The above code doesn't use `c`, which holds deprecated indices,
or `b`, which maps indices to strings.
If `crate_name::function_name` used both, it might look like this.

```json
        "b": [[0, "impl-Foo-for-Bar"]],
        "c": "OjAAAAEAAAAAAAIAEAAAABUAbgZYCQ==",
```

This attaches a disambiguator to index 1 and marks it deprecated.

The advantage of this layout is that these APIs often have implicit structure
that DEFLATE can take advantage of,
but that rustdoc can't assume.
Like how names are usually CamelCase or snake_case,
but descriptions aren't.
It also makes it easier to use a sparse data for things like boolean flags.

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

### Representing sparse columns

#### VLQ Hex

This format is, as far as I know, used nowhere other than rustdoc.
It follows this grammar:

```ebnf
VLQHex = { VHItem | VHBackref }
VHItem = VHNumber | ( '{', {VHItem}, '}' )
VHNumber = { '@' | 'A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'G' | 'H' | 'I' | 'J' | 'K' | 'L' | 'M' | 'N' | 'O' }, ( '`' | 'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'g' | 'h' | 'i' | 'j' | 'k ' | 'l' | 'm' | 'n' | 'o' )
VHBackref = ( '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | ':' | ';' | '<' | '=' | '>' | '?' )
```

A VHNumber is a variable-length, self-terminating base16 number
(terminated because the last hexit is lowercase while all others are uppercase).
The sign bit is represented using [zig-zag encoding].

This alphabet is chosen because the characters can be turned into hexits by
masking off the last four bits of the ASCII encoding.

A major feature of this encoding, as with all of the "compression" done in rustdoc,
is that it can remain in its compressed format *even in memory at runtime*.
This is why `HBackref` is only used at the top level,
and why we don't just use [Flate] for everything: the decoder in search.js
will reuse the entire decoded object whenever a backref is seen,
saving decode work and memory.

[zig-zag encoding]: https://en.wikipedia.org/wiki/Variable-length_quantity#Zigzag_encoding
[Flate]: https://en.wikipedia.org/wiki/Deflate

#### Roaring Bitmaps

Flag-style data, such as deprecation and empty descriptions,
are stored using the [standard Roaring Bitmap serialization format with runs].
The data is then base64 encoded when writing it.

As a brief overview: a roaring bitmap is a chunked array of bits,
described in [this paper].
A chunk can either be a list of integers, a bitfield, or a list of runs.
In any case, the search engine has to base64 decode it,
and read the chunk index itself,
but the payload data stays as-is.

All roaring bitmaps in rustdoc currently store a flag for each item index.
The crate is item 0, all others start at 1.

[standard Roaring Bitmap serialization format with runs]: https://github.com/RoaringBitmap/RoaringFormatSpec
[this paper]: https://arxiv.org/pdf/1603.06549.pdf

### How descriptions are stored

The largest amount of data,
and the main thing Rustdoc Search deals with that isn't
actually used for searching, is descriptions.
In a SERP table, this is what appears on the rightmost column.

> | item type | item path             | ***description*** (this part)                       |
> | --------- | --------------------- | --------------------------------------------------- |
> | function  | my_crate::my_function | This function gets the name of an integer with Data |

When someone runs a search in rustdoc for the first time, their browser will
work through a "sandwich workload" of three steps:

1. Download the search-index.js and search.js files (a network bottleneck).
2. Perform the actual search (a CPU and memory bandwidth bottleneck).
3. Download the description data (another network bottleneck).

Reducing the amount of data downloaded here will almost always increase latency,
by delaying the decision of what to download behind other work and/or adding
data dependencies where something can't be downloaded without first downloading
something else. In this case, we can't start downloading descriptions until
after the search is done, because that's what allows it to decide *which*
descriptions to download (it needs to sort the results then truncate to 200).

To do this, two columns are stored in the search index, building on both
Roaring Bitmaps and on VLQ Hex.

* `e` is an index of **e**mpty descriptions. It's a [roaring bitmap] of
  each item (the crate itself is item 0, the rest start at 1).
* `D` is a shard list, stored in [VLQ hex] as flat list of integers.
  Each integer gives you the number of descriptions in the shard.
  As the decoder walks the index, it checks if the description is empty.
  if it's not, then it's in the "current" shard. When all items are
  exhausted, it goes on to the next shard.

Inside each shard is a newline-delimited list of descriptions,
wrapped in a JSONP-style function call.

[roaring bitmap]: #roaring-bitmaps
[VLQ hex]: #vlq-hex

### `i`, `f`, and `p`

`i` and `f` both index into `p`, the array of parent items.

`i` is just a one-indexed number
(not zero-indexed because `0` is used for items that have no parent item).
It's different from `q` because `q` represents the parent *module or crate*,
which everything has,
while `i`/`q` are used for *type and trait-associated items* like methods.

`f`, the function signatures, use a [VLQ hex] tree.
A number is either a one-indexed reference into `p`,
a negative number representing a generic,
or zero for null.

(the internal object representation also uses negative numbers,
even after decoding,
to represent generics).

For example, `{{gb}{d}}` is equivalent to the json `[[3, 1], [2]]`.
Because of zigzag encoding, `` ` `` is +0, `a` is -0 (which is not used),
`b` is +1, and `c` is -1.

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

## Re-exports

[Re-export inlining] allows the same item to be found by multiple names.
Search supports this by giving the same item multiple entries and tracking a canonical path
for any items where that differs from the given path.

For example, this sample index has a single struct exported from two paths:

```json
[
    [ "crate_name", {
        "doc": "Documentation",
        "n": ["Data", "Data"],
        "t": "FF",
        "d": ["The data struct", "The data struct"],
        "q": [[0, "crate_name"], [1, "crate_name::submodule"]],
        "i": [0, 0],
        "p": [],
        "f": "``",
        "b": [],
        "c": [],
        "a": [],
        "r": [[0, 1]],
    }]
]
```

The important part of this example is the `r` array,
which indicates that path entry 1 in the `q` array is
the canonical path for item 0.
That is, `crate_name::Data` has a canonical path of `crate_name::submodule::Data`.

This might sound like a strange design, since it has the duplicate data.
It's done that way because inlining can happen across crates,
which are compiled separately and might not all be present in the docs.

```json
[
  [ "crate_name", ... ],
  [ "crate_name_2", { "q": [[0, "crate_name::submodule"], [5, "core::option"]], ... }]
]
```

In the above example, a canonical path actually comes from a dependency,
and another one comes from an inlined standard library item:
the canonical path isn't even in the index!
The canonical path might also be private.
In either case, it's never shown to the user, and is only used for deduplication.

Associated types, like methods, store them differently.
These types are connected with an entry in `p` (their "parent")
and each one has an optional third tuple element:

    "p": [[5, "Data", 0, 1]]

That's:

- 5: It's a struct
- "Data": Its name
- 0: Its display path, "crate_name"
- 1: Its canonical path, "crate_name::submodule"

In both cases, the canonical path might not be public at all,
or it might be from another crate that isn't in the docs,
so it's never shown to the user, but is used for deduplication.

[Re-export inlining]: https://doc.rust-lang.org/nightly/rustdoc/write-documentation/re-exports.html

## Testing the search engine

While the generated UI is tested using `rustdoc-gui` tests, the
primary way the search engine is tested is the `rustdoc-js` and
`rustdoc-js-std` tests. They run in NodeJS.

A `rustdoc-js` test has a `.rs` and `.js` file, with the same name.
The `.rs` file specifies the hypothetical library crate to run
the searches on (make sure you mark anything you need to find as `pub`).
The `.js` file specifies the actual searches.
The `rustdoc-js-std` tests are the same, but don't require an `.rs`
file, since they use the standard library.

The `.js` file is like a module (except the loader takes care of
`exports` for you). It uses these variables:

|      Name      |              Type              | Description
| -------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------
| `FILTER_CRATE` | `string`                       | Only include results from the given crate. In the GUI, this is the "Results in <kbd>crate</kbd>" drop-down menu.
| `EXPECTED`     | `[ResultsTable]\|ResultsTable` | List of tests to run, specifying what the hypothetical user types into the search box and sees in the tabs
| `PARSED`       | `[ParsedQuery]\|ParsedQuery`   | List of parser tests to run, without running an actual search

`FILTER_CRATE` can be left out (equivalent to searching "all crates"), but you
have to specify `EXPECTED` or `PARSED`.



By default, the test fails if any of the results specified in the test case are
not found after running the search, or if the results found after running the
search don't appear in the same order that they do in the test.
The actual search results may, however, include results that aren't in the test.
To override this, specify any of the following magic comments.
Put them on their own line, without indenting.

* `// exact-check`: If search results appear that aren't part of the test case,
  then fail.
* `// ignore-order`: Allow search results to appear in any order.
* `// should-fail`: Used to write negative tests.

Standard library tests usually shouldn't specify `// exact-check`, since we
want the libs team to be able to add new items without causing unrelated
tests to fail, but standalone tests will use it more often.

The `ResultsTable` and `ParsedQuery` types are specified in
[`rustdoc.d.ts`](https://github.com/rust-lang/rust/blob/master/src/librustdoc/html/static/js/rustdoc.d.ts).

For example, imagine we needed to fix a bug where a function named
`constructor` couldn't be found. To do this, write two files:

```rust
// tests/rustdoc-js/constructor_search.rs
// The test case needs to find this result.
pub fn constructor(_input: &str) -> i32 { 1 }
```

```js
// tests/rustdoc-js/constructor_search.js
// exact-check
// Since this test runs against its own crate,
// new items should not appear in the search results.
const EXPECTED = [
  // This first test targets name-based search.
  {
    query: "constructor",
    others: [
      { path: "constructor_search", name: "constructor" },
    ],
    in_args: [],
    returned: [],
  },
  // This test targets the second tab.
  {
    query: "str",
    others: [],
    in_args: [
      { path: "constructor_search", name: "constructor" },
    ],
    returned: [],
  },
  // This test targets the third tab.
  {
    query: "i32",
    others: [],
    in_args: [],
    returned: [
      { path: "constructor_search", name: "constructor" },
    ],
  },
  // This test targets advanced type-driven search.
  {
    query: "str -> i32",
    others: [
      { path: "constructor_search", name: "constructor" },
    ],
    in_args: [],
    returned: [],
  },
]
```