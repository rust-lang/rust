An inherent `impl` was written on a dyn auto trait.

Erroneous code example:

```compile_fail,E0785
#![feature(auto_traits)]

auto trait AutoTrait {}

impl dyn AutoTrait {}
```

Dyn objects allow any number of auto traits, plus at most one non-auto trait.
The non-auto trait becomes the "principal trait".

When checking if an impl on a dyn trait is coherent, the principal trait is
normally the only one considered. Since the erroneous code has no principal
trait, it cannot be implemented at all.

Working example:

```
#![feature(auto_traits)]

trait PrincipalTrait {}

auto trait AutoTrait {}

impl dyn PrincipalTrait + AutoTrait + Send {}
```
