The matched value was assigned in a match guard.

Erroneous code example:

```compile_fail,E0510
let mut x = Some(0);
match x {
    None => {}
    Some(_) if { x = None; false } => {} // error!
    Some(_) => {}
}
```

When matching on a variable it cannot be mutated in the match guards, as this
could cause the match to be non-exhaustive.

Here executing `x = None` would modify the value being matched and require us
to go "back in time" to the `None` arm. To fix it, change the value in the match
arm:

```
let mut x = Some(0);
match x {
    None => {}
    Some(_) => {
        x = None; // ok!
    }
}
```
