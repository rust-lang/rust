An undeclared label was used.

Erroneous code example:

```compile_fail,E0426
loop {
    break 'a; // error: use of undeclared label `'a`
}
```

Please verify you spelled or declared the label correctly. Example:

```
'a: loop {
    break 'a; // ok!
}
```
