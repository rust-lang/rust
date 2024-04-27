The entry point of the program was marked as `async`.

Erroneous code example:

```compile_fail,E0752
async fn main() -> Result<(), ()> { // error!
    Ok(())
}
```

`fn main()` or the specified start function is not allowed to be `async`. Not
having a correct async runtime library setup may cause this error. To fix it,
declare the entry point without `async`:

```
fn main() -> Result<(), ()> { // ok!
    Ok(())
}
```
