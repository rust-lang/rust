- Feature Name: entry_v3
- Start Date: 2015-03-01
- RFC PR: https://github.com/rust-lang/rfcs/pull/921
- Rust Issue: https://github.com/rust-lang/rust/issues/23508

# Summary

Replace `Entry::get` with `Entry::or_insert` and
`Entry::or_insert_with` for better ergonomics and clearer code.

# Motivation

Entry::get was introduced to reduce a lot of the boiler-plate involved in simple Entry usage. Two
incredibly common patterns in particular stand out:

```
match map.entry(key) => {
    Entry::Vacant(entry) => { entry.insert(1); },
    Entry::Occupied(entry) => { *entry.get_mut() += 1; },
}
```

```
match map.entry(key) => {
    Entry::Vacant(entry) => { entry.insert(vec![val]); },
    Entry::Occupied(entry) => { entry.get_mut().push(val); },
}
```

This code is noisy, and is visibly fighting the Entry API a bit, such as having to supress
the return value of insert. It requires the `Entry` enum to be imported into scope. It requires
the user to learn a whole new API. It also introduces a "many ways to do it" stylistic ambiguity:

```
match map.entry(key) => {
    Entry::Vacant(entry) => entry.insert(vec![]),
    Entry::Occupied(entry) => entry.into_mut(),
}.push(val);
```

Entry::get tries to address some of this by doing something similar to `Result::ok`.
It maps the Entry into a more familiar Result, while automatically converting the
Occupied case into an `&mut V`. Usage looks like:


```
*map.entry(key).get().unwrap_or_else(|entry| entry.insert(0)) += 1;
```

```
map.entry(key).get().unwrap_or_else(|entry| entry.insert(vec![])).push(val);
```

This is certainly *nicer*. No imports are needed, the Occupied case is handled, and we're closer
to a "only one way". However this is still fairly tedious and arcane. `get` provides little
meaning for what is done; unwrap_or_else is long and scary-sounding; and VacantEntry litterally
*only* supports `insert`, so having to call it seems redundant.

# Detailed design

Replace `Entry::get` with the following two methods:

```
    /// Ensures a value is in the entry by inserting the default if empty, and returns
    /// a mutable reference to the value in the entry.
    pub fn or_insert(self. default: V) -> &'a mut V {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert(default),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default function if empty,
    /// and returns a mutable reference to the value in the entry.
    pub fn or_insert_with<F: FnOnce() -> V>(self. default: F) -> &'a mut V {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert(default()),
        }
    }
```

which allows the following:


```
*map.entry(key).or_insert(0) += 1;
```

```
// vec![] doesn't even allocate, and is only 3 ptrs big.
map.entry(key).or_insert(vec![]).push(val);
```

```
let val = map.entry(key).or_insert_with(|| expensive(big, data));
```

Look at all that ergonomics. *Look at it*. This pushes us more into the "one right way"
territory, since this is unambiguously clearer and easier than a full `match` or abusing Result.
Novices don't really need to learn the entry API at all with this. They can just learn the
`.entry(key).default(value)` incantation to start, and work their way up to more complex
usage later.

Oh hey look this entire RFC is already implemented with all of `rust-lang/rust`'s `entry`
usage audited and updated: https://github.com/rust-lang/rust/pull/22930

# Drawbacks

Replaces the composability of just mapping to a Result with more adhoc specialty methods. This
is hardly a drawback for the reasons stated in the RFC. Maybe someone was really leveraging
the Result-ness in an exotic way, but it was likely an abuse of the API. Regardless, the `get`
method is trivial to write as a consumer of the API.

# Alternatives

Settle for `Result` chumpsville or abandon this sugar altogether. Truly, fates worse than death.

# Unresolved questions

None.
