A non-ASCII identifier was used in an invalid context.

Erroneous code examples:

```compile_fail,E0754

mod řųśť; // error!

#[no_mangle]
fn řųśť() {} // error!

fn main() {}
```

Non-ASCII can be used as module names if it is inlined or if a `#[path]`
attribute is specified. For example:

```
mod řųśť { // ok!
    const IS_GREAT: bool = true;
}

fn main() {}
```
