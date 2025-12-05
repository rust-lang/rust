A return statement was found outside of a function body.

Erroneous code example:

```compile_fail,E0572
const FOO: u32 = return 0; // error: return statement outside of function body

fn main() {}
```

To fix this issue, just remove the return keyword or move the expression into a
function. Example:

```
const FOO: u32 = 0;

fn some_fn() -> u32 {
    return FOO;
}

fn main() {
    some_fn();
}
```
