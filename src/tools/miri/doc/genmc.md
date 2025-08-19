# **(WIP)** Documentation for Miri-GenMC

[GenMC](https://github.com/MPI-SWS/genmc) is a stateless model checker for exploring concurrent executions of a program.
Miri-GenMC integrates that model checker into Miri.

**NOTE: Currently, no actual GenMC functionality is part of Miri, this is still WIP.**

<!-- FIXME(genmc): add explanation. -->

## Usage

**IMPORTANT: The license of GenMC and thus the `genmc-sys` crate in the Miri repo is currently "GPL-3.0-or-later", so a binary produced with the `genmc` feature is subject to the requirements of the GPL. As long as that remains the case, the `genmc` feature of Miri is OFF-BY-DEFAULT and must be OFF for all Miri releases.**

For testing/developing Miri-GenMC (while keeping in mind the licensing issues):
- clone the Miri repo.
- build Miri-GenMC with `./miri build --features=genmc`.
- OR: install Miri-GenMC in the current system with `./miri install --features=genmc`

Basic usage:
```shell
MIRIFLAGS="-Zmiri-genmc" cargo miri run
```

<!-- FIXME(genmc): explain options. -->

<!-- FIXME(genmc): explain Miri-GenMC specific functions. -->

## Tips

<!-- FIXME(genmc): add tips for using Miri-GenMC more efficiently. -->

## Limitations

Some or all of these limitations might get removed in the future:

- Borrow tracking is currently incompatible (stacked/tree borrows).
- Only Linux is supported for now.
- No support for 32-bit or big-endian targets.
- No cross-target interpretation.

<!-- FIXME(genmc): document remaining limitations -->

## Development

GenMC is written in C++, which complicates development a bit.
The prerequisites for building Miri-GenMC are:
- A compiler with C++23 support.
- LLVM developments headers and clang.
  <!-- FIXME(genmc,llvm): remove once LLVM dependency is no longer required. -->

The actual code for GenMC is not contained in the Miri repo itself, but in a [separate GenMC repo](https://github.com/MPI-SWS/genmc) (with its own maintainers).
These sources need to be available to build Miri-GenMC.
The process for obtaining them is as follows:
- By default, a fixed commit of GenMC is downloaded to `genmc-sys/genmc-src` and built automatically.
  (The commit is determined by `GENMC_COMMIT` in `genmc-sys/build.rs`.)
- If you want to overwrite that, set the `GENMC_SRC_PATH` environment variable to a path that contains the GenMC sources.
  If you place this directory inside the Miri folder, it is recommended to call it `genmc-src` as that tells `./miri fmt` to avoid
  formatting the Rust files inside that folder.

<!-- FIXME(genmc): explain how submitting code to GenMC should be handled. -->

<!-- FIXME(genmc): explain development. -->
