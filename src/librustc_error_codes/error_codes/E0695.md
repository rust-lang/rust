A `break` statement without a label appeared inside a labeled block.

Erroneous code example:

```compile_fail,E0695
# #![feature(label_break_value)]
loop {
    'a: {
        break;
    }
}
```

Make sure to always label the `break`:

```
# #![feature(label_break_value)]
'l: loop {
    'a: {
        break 'l;
    }
}
```

Or if you want to `break` the labeled block:

```
# #![feature(label_break_value)]
loop {
    'a: {
        break 'a;
    }
    break;
}
```
