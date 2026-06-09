Tests that specialization is working correctly:

- Dispatch
  - [On methods](specialization-basics.rs), includes:
    - Specialization via adding a trait bound
      - Including both remote and local traits
    - Specialization via pure structure (e.g. `(T, U)` vs `(T, T)`)
    - Specialization via concrete types vs unknown types
      - In top level of the trait reference
      - Embedded within another type (`Vec<T>` vs `Vec<i32>`)
  - [Specialization based on super trait relationships](specialization-supertraits.rs)
  - [On assoc fns](specialization-assoc-fns.rs)
  - [Ensure that impl order doesn't matter](specialization-out-of-order.rs)

- Item inheritance
  - [Correct default cascading for methods](specialization-default-methods.rs)
  - Inheritance works across impls with varying generics
    - [With projections](specialization-translate-projections.rs)
    - [With projections that involve input types](specialization-translate-projections-with-params.rs)

- Normalization issues
  - [Non-default assoc types can be projected](specialization-projection.rs)
    - Including non-specialized cases
    - Including specialized cases
  - [Specialized Impls can happen on projections](specialization-on-projection.rs)
  - [Projections and aliases play well together](specialization-projection-alias.rs)
  - [Projections involving specialization allowed in the trait ref for impls, and overlap can still be determined](specialization-overlap-projection.rs)
    - Only works for the simple case where the most specialized impl directly
      provides a non-`default` associated type

- Across crates
  - [For traits defined in upstream crate](specialization-allowed-cross-crate.rs)
  - [Full method dispatch tests, drawing from upstream crate](specialization-cross-crate.rs)
    - Including *additional* local specializations
  - [Full method dispatch tests, *without* turning on specialization in local crate](specialization-cross-crate-no-gate.rs)
  - [Test that defaults cascade correctly from upstream crates](specialization-cross-crate-defaults.rs)
    - Including *additional* local use of defaults
