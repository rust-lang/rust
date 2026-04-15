# Contributing to ThingOS

ThingOS is maintained as a fork of `rust-lang/rust`, but it is not the Rust project.

Contributions should preserve two constraints:

* advance the ThingOS model and implementation
* keep the tree mergeable with upstream Rust where practical

If a change is ThingOS-specific, prefer keeping it narrowly scoped and clearly named so upstream sync work stays mechanical.

If a change is a general Rust bug fix or toolchain improvement, consider whether it should also be proposed upstream.

Repository procedures, review policy, and contribution workflow are defined by the current ThingOS maintainers rather than Rust project policy.

## Conceptual model and naming conventions

ThingOS is migrating from a Unix/Linux Process+Thread model toward a set of
first-class concepts: **Task**, **Job**, **Space**, **Authority**, **Place**,
and **Group**.  Before writing or reviewing code that touches `kernel/`, `abi/`,
`bran/`, `stem/`, or `userspace/`, consult:

- [`docs/migration/concept-mapping.md`](docs/migration/concept-mapping.md) —
  canonical mapping between legacy Unix concepts and ThingOS/ThingOS concepts,
  including naming rules and migration guidance.
- [`docs/migration/review-guidelines.md`](docs/migration/review-guidelines.md) —
  PR review checklist derived from the concept mapping.
