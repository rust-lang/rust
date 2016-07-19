## `return` [RFC #968]

Terminate `return` statements with semicolons:

``` rust,ignore
fn foo(bar: i32) -> Option<i32> {
    if some_condition() {
        return None;
    }

    ...
}
```
