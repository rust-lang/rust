When matching against a range, the compiler verifies that the range is
non-empty. Range patterns include both end-points, so this is equivalent to
requiring the start of the range to be less than or equal to the end of the
range.

Erroneous code example:

```compile_fail,E0030
match 5u32 {
    // This range is ok, albeit pointless.
    1 ..= 1 => {}
    // This range is empty, and the compiler can tell.
    1000 ..= 5 => {}
}
```
