An item which isn't a unit struct, a variant, nor a constant has been used as a
match pattern.

Erroneous code example:

```compile_fail,E0533
struct Tortoise;

impl Tortoise {
    fn turtle(&self) -> u32 { 0 }
}

match 0u32 {
    Tortoise::turtle => {} // Error!
    _ => {}
}
if let Tortoise::turtle = 0u32 {} // Same error!
```

If you want to match against a value returned by a method, you need to bind the
value first:

```
struct Tortoise;

impl Tortoise {
    fn turtle(&self) -> u32 { 0 }
}

match 0u32 {
    x if x == Tortoise.turtle() => {} // Bound into `x` then we compare it!
    _ => {}
}
```
