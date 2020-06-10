The `start` function was defined with a where clause.

Erroneous code example:

```compile_fail,E0647
#![feature(start)]

#[start]
fn start(_: isize, _: *const *const u8) -> isize where (): Copy {
    //^ error: start function is not allowed to have a where clause
    0
}
```
