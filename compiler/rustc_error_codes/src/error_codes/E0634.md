A type has conflicting `packed` representation hints.

Erroneous code examples:

```compile_fail,E0634
#[repr(packed, packed(2))] // error!
struct Company(i32);

#[repr(packed(2))] // error!
#[repr(packed)]
struct Company(i32);
```

You cannot use conflicting `packed` hints on a same type. If you want to pack a
type to a given size, you should provide a size to packed:

```
#[repr(packed)] // ok!
struct Company(i32);
```
