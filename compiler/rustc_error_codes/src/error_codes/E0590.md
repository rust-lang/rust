`break` or `continue` keywords were used in a condition of a `while` loop
without a label.

Erroneous code code:

```compile_fail,E0590
while break {}
```

`break` or `continue` must include a label when used in the condition of a
`while` loop.

To fix this, add a label specifying which loop is being broken out of:

```
'foo: while break 'foo {}
```
