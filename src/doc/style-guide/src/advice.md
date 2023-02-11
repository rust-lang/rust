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

 * Types shall be `UpperCamelCase`,
 * Enum variants shall be `UpperCamelCase`,
 * Struct fields shall be `snake_case`,
 * Function and method names shall be `snake_case`,
 * Local variables shall be `snake_case`,
 * Macro names shall be `snake_case`,
 * Constants (`const`s and immutable `static`s) shall be `SCREAMING_SNAKE_CASE`.
 * When a name is forbidden because it is a reserved word (e.g., `crate`), use a
   trailing underscore to make the name legal (e.g., `crate_`), or use raw
   identifiers if possible.

### Modules

Avoid `#[path]` annotations where possible.
