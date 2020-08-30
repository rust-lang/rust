A double quote string (`"`) was not terminated.

Erroneous code example:

```compile_fail,E0765
let s = "; // error!
```

To fix this error, add the missing double quote at the end of the string:

```
let s = ""; // ok!
```
