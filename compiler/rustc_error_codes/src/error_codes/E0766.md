A double quote byte string (`b"`) was not terminated.

Erroneous code example:

```compile_fail,E0766
let s = b"; // error!
```

To fix this error, add the missing double quote at the end of the string:

```
let s = b""; // ok!
```
