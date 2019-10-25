# Assists

## `convert_to_guarded_return`

Replace a large conditional with a guarded return.

```rust
// BEFORE
fn main() {
    <|>if cond {
        foo();
        bar();
    }
}

// AFTER
fn main() {
    if !cond {
        return;
    }
    foo();
    bar();
}
```
