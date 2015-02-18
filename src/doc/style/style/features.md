## `return` [FIXME: needs RFC]

Terminate `return` statements with semicolons:

``` rust
fn foo(bar: int) -> Option<int> {
    if some_condition() {
        return None;
    }

    ...
}
```
