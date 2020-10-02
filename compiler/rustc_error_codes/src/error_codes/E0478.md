A lifetime bound was not satisfied.

Erroneous code example:

```compile_fail,E0478
// Check that the explicit lifetime bound (`'SnowWhite`, in this example) must
// outlive all the superbounds from the trait (`'kiss`, in this example).

trait Wedding<'t>: 't { }

struct Prince<'kiss, 'SnowWhite> {
    child: Box<Wedding<'kiss> + 'SnowWhite>,
    // error: lifetime bound not satisfied
}
```

In this example, the `'SnowWhite` lifetime is supposed to outlive the `'kiss`
lifetime but the declaration of the `Prince` struct doesn't enforce it. To fix
this issue, you need to specify it:

```
trait Wedding<'t>: 't { }

struct Prince<'kiss, 'SnowWhite: 'kiss> { // You say here that 'SnowWhite
                                          // must live longer than 'kiss.
    child: Box<Wedding<'kiss> + 'SnowWhite>, // And now it's all good!
}
```
