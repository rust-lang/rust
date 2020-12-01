A function is using `continue` keyword incorrectly.

Erroneous code example:

```compile_fail,E0696
fn continue_simple() {
    'b: {
        continue; // error!
    }
}
fn continue_labeled() {
    'b: {
        continue 'b; // error!
    }
}
fn continue_crossing() {
    loop {
        'b: {
            continue; // error!
        }
    }
}
```

Here we have used the `continue` keyword incorrectly. As we
have seen above that `continue` pointing to a labeled block.

To fix this we have to use the labeled block properly.
For example:

```
fn continue_simple() {
    'b: loop {
        continue ; // ok!
    }
}
fn continue_labeled() {
    'b: loop {
        continue 'b; // ok!
    }
}
fn continue_crossing() {
    loop {
        'b: loop {
            continue; // ok!
        }
    }
}
```
