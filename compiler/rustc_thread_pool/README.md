Note: This is an unstable fork made for use in rustc

Rayon-core represents the "core, stable" APIs of Rayon: join, scope, and so forth, as well as the ability to create custom thread-pools with ThreadPool.

Maybe worth mentioning: users are not necessarily intended to directly access rayon-core; all its APIs are mirrored in the rayon crate. To that end, the examples in the docs use rayon::join and so forth rather than rayon_core::join.

rayon-core aims to never, or almost never, have a breaking change to its API, because each revision of rayon-core also houses the global thread-pool (and hence if you have two simultaneous versions of rayon-core, you have two thread-pools).

Please see [Rayon Docs] for details about using Rayon.

[Rayon Docs]: https://docs.rs/rayon/

Rayon-core currently requires `rustc 1.63.0` or greater.
