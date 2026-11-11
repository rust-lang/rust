An invalid left-hand side expression was used on an assignment operation.

Erroneous code example:

```compile_fail,E0067
12 += 1; // error!
```

You need to have a place expression to be able to assign it something. For
example:

```
let mut x: i8 = 12;
x += 1; // ok!
```
