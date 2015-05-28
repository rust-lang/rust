## `return` [FIXME: needs RFC]

Terminate `return` statements with semicolons:

``` rust
fn foo(bar: i32) -> Option<i32> {
    if some_condition() {
        return None;
    }

    ...
}
```
