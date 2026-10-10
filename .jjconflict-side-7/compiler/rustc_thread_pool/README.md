Note: This is an unstable fork made for use in rustc

Rayon-core represents the "core, stable" APIs of Rayon: join, scope, and so forth, as well as the ability to create custom thread-pools with ThreadPool.

Maybe worth mentioning: users are not necessarily intended to directly access rustc_thread_pool; all its APIs are mirrored in the rayon crate. To that end, the examples in the docs use rayon::join and so forth rather than rayon_core::join.


Please see [Rayon Docs] for details about using Rayon.

[Rayon Docs]: https://docs.rs/rayon/
