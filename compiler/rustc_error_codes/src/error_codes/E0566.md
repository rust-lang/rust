Conflicting representation hints have been used on a same item.

Erroneous code example:

```compile_fail,E0566
#[repr(u32, u64)]
enum Repr { A }
```

In most cases (if not all), using just one representation hint is more than
enough. If you want to have a representation hint depending on the current
architecture, use `cfg_attr`. Example:

```
#[cfg_attr(linux, repr(u32))]
#[cfg_attr(not(linux), repr(u64))]
enum Repr { A }
```
