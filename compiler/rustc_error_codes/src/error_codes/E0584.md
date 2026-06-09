A doc comment that is not attached to anything has been encountered.

Erroneous code example:

```compile_fail,E0584
trait Island {
    fn lost();

    /// I'm lost!
}
```

A little reminder: a doc comment has to be placed before the item it's supposed
to document. So if you want to document the `Island` trait, you need to put a
doc comment before it, not inside it. Same goes for the `lost` method: the doc
comment needs to be before it:

```
/// I'm THE island!
trait Island {
    /// I'm lost!
    fn lost();
}
```
