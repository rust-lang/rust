% Post-1.0 changes

### Higher-kinded types

* A trait encompassing both `Iterable<T>` for some fixed `T` and
  `FromIterator<U>` for _all_ `U` (where HKT comes in).  The train
  could provide e.g. a default `map` method producing the same kind of
  the container, but with a new type parameter.

* **Monadic-generic programming**? Can we add this without deprecating
  huge swaths of the API (including `Option::map`, `option::collect`,
  `result::collect`, `try!` etc.
