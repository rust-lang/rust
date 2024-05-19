# `instrument-xray`

The tracking issue for this feature is: [#102921](https://github.com/rust-lang/rust/issues/102921).

------------------------

Enable generation of NOP sleds for XRay function tracing instrumentation.
For more information on XRay,
read [LLVM documentation](https://llvm.org/docs/XRay.html),
and/or the [XRay whitepaper](http://research.google.com/pubs/pub45287.html).

Set the `-Z instrument-xray` compiler flag in order to enable XRay instrumentation.

  - `-Z instrument-xray` – use the default settings
  - `-Z instrument-xray=skip-exit` – configure a custom setting
  - `-Z instrument-xray=ignore-loops,instruction-threshold=300` –
    multiple settings separated by commas

Supported options:

  - `always` – force instrumentation of all functions
  - `never` – do no instrument any functions
  - `ignore-loops` – ignore presence of loops,
    instrument functions based only on instruction count
  - `instruction-threshold=10` – set a different instruction threshold for instrumentation
  - `skip-entry` – do no instrument function entry
  - `skip-exit` – do no instrument function exit

The default settings are:

  - instrument both entry & exit from functions
  - instrument functions with at least 200 instructions,
    or containing a non-trivial loop

Note that `-Z instrument-xray` only enables generation of NOP sleds
which on their own don't do anything useful.
In order to actually trace the functions,
you will need to link a separate runtime library of your choice,
such as Clang's [XRay Runtime Library](https://www.llvm.org/docs/XRay.html#xray-runtime-library).
