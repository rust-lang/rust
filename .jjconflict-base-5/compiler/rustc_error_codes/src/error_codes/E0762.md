A character literal wasn't ended with a quote.

Erroneous code example:

```compile_fail,E0762
static C: char = '●; // error!
```

To fix this error, add the missing quote:

```
static C: char = '●'; // ok!
```
