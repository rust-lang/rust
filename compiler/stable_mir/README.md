This crate is currently developed in-tree together with the compiler.

Our goal is to start publishing `stable_mir` into crates.io.
Until then, users will use this as any other rustc crate, via extern crate.

## Stable MIR Design

The stable-mir will follow a similar approach to proc-macro2. Its
implementation is split between two main crates:

- `stable_mir`: Public crate, to be published on crates.io, which will contain
the stable data structure as well as calls to `rustc_smir` APIs and
translation between stable and internal constructs.
- `rustc_smir`: This crate implements the public APIs to the compiler.
It is responsible for gathering all the information requested, and providing
the data in its unstable form.

I.e.,
tools will depend on `stable_mir` crate,
which will invoke the compiler using APIs defined in `rustc_smir`.

I.e.:

```
    ┌──────────────────────────────────┐           ┌──────────────────────────────────┐
    │   External Tool     ┌──────────┐ │           │ ┌──────────┐   Rust Compiler     │
    │                     │          │ │           │ │          │                     │
    │                     │stable_mir| │           │ │rustc_smir│                     │
    │                     │          │ ├──────────►| │          │                     │
    │                     │          │ │◄──────────┤ │          │                     │
    │                     │          │ │           │ │          │                     │
    │                     │          │ │           │ │          │                     │
    │                     └──────────┘ │           │ └──────────┘                     │
    └──────────────────────────────────┘           └──────────────────────────────────┘
```

More details can be found here:
https://hackmd.io/XhnYHKKuR6-LChhobvlT-g?view
