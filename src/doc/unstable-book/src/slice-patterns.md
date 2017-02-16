# `slice_patterns`

The tracking issue for this feature is: [#23121]

[#23121]: https://github.com/rust-lang/rust/issues/23121

See also [`advanced_slice_patterns`](advanced-slice-patterns.html).

------------------------


If you want to match against a slice or array, you can use `&` with the
`slice_patterns` feature:

```rust
#![feature(slice_patterns)]

fn main() {
    let v = vec!["match_this", "1"];

    match &v[..] {
        &["match_this", second] => println!("The second element is {}", second),
        _ => {},
    }
}
```

