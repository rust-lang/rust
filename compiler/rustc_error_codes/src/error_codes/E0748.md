A raw string isn't correctly terminated because the trailing `#` count doesn't
match its leading `#` count.

Erroneous code example:

```compile_fail,E0748
let dolphins = r##"Dolphins!"#; // error!
```

To terminate a raw string, you have to have the same number of `#` at the end
as at the beginning. Example:

```
let dolphins = r#"Dolphins!"#; // One `#` at the beginning, one at the end so
                               // all good!
```
