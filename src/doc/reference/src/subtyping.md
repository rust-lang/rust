# Subtyping

Subtyping is implicit and can occur at any stage in type checking or
inference. Subtyping in Rust is very restricted and occurs only due to
variance with respect to lifetimes and between types with higher ranked
lifetimes. If we were to erase lifetimes from types, then the only subtyping
would be due to type equality.

Consider the following example: string literals always have `'static`
lifetime. Nevertheless, we can assign `s` to `t`:

```
fn bar<'a>() {
    let s: &'static str = "hi";
    let t: &'a str = s;
}
```
Since `'static` "lives longer" than `'a`, `&'static str` is a subtype of
`&'a str`.
