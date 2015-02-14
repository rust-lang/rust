% Convenience methods

### Provide small, coherent sets of convenience methods. **[FIXME: needs RFC]**

_Convenience methods_ wrap up existing functionality in a more convenient
way. The work done by a convenience method varies widely:

* _Re-providing functions as methods_. For example, the `std::path::Path` type
  provides methods like `stat` on `Path`s that simply invoke the corresponding
  function in `std::io::fs`.
* _Skipping through conversions_. For example, the `str` type provides a
  `.len()` convenience method which is also expressible as `.as_bytes().len()`.
  Sometimes the conversion is more complex: the `str` module also provides
  `from_chars`, which encapsulates a simple use of iterators.
* _Encapsulating common arguments_. For example, vectors of `&str`s
  provide a `connect` as well as a special case, `concat`, that is expressible
  using `connect` with a fixed separator of `""`.
* _Providing more efficient special cases_. The `connect` and `concat` example
  also applies here: singling out `concat` as a special case allows for a more
  efficient implementation.

  Note, however, that the `connect` method actually detects the special case
  internally and invokes `concat`. Usually, it is not necessary to add a public
  convenience method just for efficiency gains; there should also be a
  _conceptual_ reason to add it, e.g. because it is such a common special case.

It is tempting to add convenience methods in a one-off, haphazard way as
common use patterns emerge. Avoid this temptation, and instead _design_ small,
coherent sets of convenience methods that are easy to remember:

* _Small_: Avoid combinatorial explosions of convenience methods. For example,
  instead of adding `_str` variants of methods that provide a `str` output,
  instead ensure that the normal output type of methods is easily convertible to
  `str`.
* _Coherent_: Look for small groups of convenience methods that make sense to
  include together. For example, the `Path` API mentioned above includes a small
  selection of the most common filesystem operations that take a `Path`
  argument.  If one convenience method strongly suggests the existence of others,
  consider adding the whole group.
* _Memorable_: It is not worth saving a few characters of typing if you have to
  look up the name of a convenience method every time you use it. Add
  convenience methods with names that are obvious and easy to remember, and add
  them for the most common or painful use cases.
