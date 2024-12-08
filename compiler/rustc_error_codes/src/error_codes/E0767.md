An unreachable label was used.

Erroneous code example:

```compile_fail,E0767
'a: loop {
    || {
        loop { break 'a } // error: use of unreachable label `'a`
    };
}
```

Ensure that the label is within scope. Labels are not reachable through
functions, closures, async blocks or modules. Example:

```
'a: loop {
    break 'a; // ok!
}
```
