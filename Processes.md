This document outlines processes regarding management of rustfmt.

# Stabilising an Option

In this Section, we describe how to stabilise an option of the rustfmt's configration.

## Conditions

- The option is well tested, both in unit tests and, optimally, in real usage.
- There is no open bug about the option that prevents its use.

## Steps

Open a pull request that closes the tracking issue. The tracking issue is listed beside the option in `Configurations.md`.

- Update the `Config` enum marking the option as stable.
- Update the the `Configuration.md` file marking the option as stable.

## After the stabilisation

The option should remain backward-compatible with previous parameters of the option. For instance, if the option is an enum `enum Foo { Alice, Bob }` and the variant `Foo::Bob` is removed/renamed, existing use of the `Foo::Bob` variant should map to the new logic. Breaking changes can be applied under the condition they are version-gated.
