Multiple types were used as bounds for a closure or trait object.

Erroneous code example:

```compile_fail,E0225
fn main() {
    let _: Box<dyn std::io::Read + std::io::Write>;
}
```

Rust does not currently support this.

Auto traits such as Send and Sync are an exception to this rule:
It's possible to have bounds of one non-builtin trait, plus any number of
auto traits. For example, the following compiles correctly:

```
fn main() {
    let _: Box<dyn std::io::Read + Send + Sync>;
}
```
