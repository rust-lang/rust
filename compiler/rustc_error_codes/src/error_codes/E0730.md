An array without a fixed length was pattern-matched.

Erroneous code example:

```compile_fail,E0730
fn is_123<const N: usize>(x: [u32; N]) -> bool {
    match x {
        [1, 2, ..] => true, // error: cannot pattern-match on an
                            //        array without a fixed length
        _ => false
    }
}
```

To fix this error, you have two solutions:
 1. Use an array with a fixed length.
 2. Use a slice.

Example with an array with a fixed length:

```
fn is_123(x: [u32; 3]) -> bool { // We use an array with a fixed size
    match x {
        [1, 2, ..] => true, // ok!
        _ => false
    }
}
```

Example with a slice:

```
fn is_123(x: &[u32]) -> bool { // We use a slice
    match x {
        [1, 2, ..] => true, // ok!
        _ => false
    }
}
```
