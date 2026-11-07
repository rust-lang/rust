A multi-line (doc-)comment is unterminated.

Erroneous code example:

```compile_fail,E0758
/* I am not terminated!
```

The same goes for doc comments:

```compile_fail,E0758
/*! I am not terminated!
```

You need to end your multi-line comment with `*/` in order to fix this error:

```
/* I am terminated! */
/*! I am also terminated! */
```
