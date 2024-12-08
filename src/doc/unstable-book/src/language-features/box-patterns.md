# `box_patterns`

The tracking issue for this feature is: [#29641]

[#29641]: https://github.com/rust-lang/rust/issues/29641

------------------------

Box patterns let you match on `Box<T>`s:


```rust
#![feature(box_patterns)]

fn main() {
    let b = Some(Box::new(5));
    match b {
        Some(box n) if n < 0 => {
            println!("Box contains negative number {n}");
        },
        Some(box n) if n >= 0 => {
            println!("Box contains non-negative number {n}");
        },
        None => {
            println!("No box");
        },
        _ => unreachable!()
    }
}
```
