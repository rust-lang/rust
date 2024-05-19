A byte constant wasn't correctly ended.

Erroneous code example:

```compile_fail,E0763
let c = b'a; // error!
```

To fix this error, add the missing quote:

```
let c = b'a'; // ok!
```
