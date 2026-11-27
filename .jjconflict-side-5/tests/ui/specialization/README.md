This directory contains the test for incorrect usage of specialization that
should lead to compile failure. Those tests break down into a few categories:

- Feature gating
  - [On use of the `default` keyword](specialization-feature-gate-default.rs)
  - [On overlapping impls](specialization-feature-gate-overlap.rs)

- Overlap checking with specialization enabled
  - [Basic overlap scenarios](specialization-overlap.rs)
    - Includes purely structural overlap
    - Includes purely trait-based overlap
    - Includes mix
  - [Overlap with differing polarity](specialization-overlap-negative.rs)

- [Attempt to specialize without using `default`](specialization-no-default.rs)

- [Attempt to change impl polarity in a specialization](specialization-polarity.rs)

- Attempt to rely on projection of a `default` type
  - [Rely on it externally in both generic and monomorphic contexts](specialization-default-projection.rs)
  - [Rely on it both within an impl and outside it](specialization-default-types.rs)
