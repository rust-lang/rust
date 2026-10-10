A negative impl was made default impl.

Erroneous code example:

```compile_fail,E0750
# #![feature(negative_impls)]
# #![feature(specialization)]
trait MyTrait {
    type Foo;
}

default impl !MyTrait for u32 {} // error!
# fn main() {}
```

Negative impls cannot be default impls. A default impl supplies default values
for the items within to be used by other impls, whereas a negative impl declares
that there are no other impls. Combining it does not make sense.
