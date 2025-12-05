# Other style advice

## Expressions

Prefer to use Rust's expression oriented nature where possible;

```rust
// use
let x = if y { 1 } else { 0 };
// not
let x;
if y {
    x = 1;
} else {
    x = 0;
}
```

## Names

- Types shall be `UpperCamelCase`,
- Enum variants shall be `UpperCamelCase`,
- Struct fields shall be `snake_case`,
- Function and method names shall be `snake_case`,
- Local variables shall be `snake_case`,
- Macro names shall be `snake_case`,
- Constants (`const`s and immutable `static`s) shall be `SCREAMING_SNAKE_CASE`.
- When a name is forbidden because it is a reserved word (such as `crate`),
  either use a raw identifier (`r#crate`) or use a trailing underscore
  (`crate_`). Don't misspell the word (`krate`).

### Modules

Avoid `#[path]` annotations where possible.
