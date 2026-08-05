#### Note: this error code is no longer emitted by the compiler.

This error indicates that the bindings in a match arm would require a value to
be moved into more than one location, thus violating unique ownership. Code
like the following is invalid as it requires the entire `Option<String>` to be
moved into a variable called `op_string` while simultaneously requiring the
inner `String` to be moved into a variable called `s`.

Erroneous code example:

```compile_fail,E0382
#![feature(bindings_after_at)]

let x = Some("s".to_string());

match x {
    op_string @ Some(s) => {}, // error: use of moved value
    None => {},
}
```

See also the error E0303.
