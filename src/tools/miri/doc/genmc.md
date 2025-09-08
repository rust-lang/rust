# **(WIP)** Documentation for Miri-GenMC

[GenMC](https://github.com/MPI-SWS/genmc) is a stateless model checker for exploring concurrent executions of a program.
Miri-GenMC integrates that model checker into Miri.

**NOTE: Currently, no actual GenMC functionality is part of Miri, this is still WIP.**

<!-- FIXME(genmc): add explanation. -->

## Usage

For testing/developing Miri-GenMC:
- clone the Miri repo.
- build Miri-GenMC with `./miri build --features=genmc`.
- OR: install Miri-GenMC in the current system with `./miri install --features=genmc`

Basic usage:
```shell
MIRIFLAGS="-Zmiri-genmc" cargo miri run
```

Note that `cargo miri test` in GenMC mode is currently not supported.

### Supported Parameters

- `-Zmiri-genmc`: Enable GenMC mode (not required if any other GenMC options are used).
- `-Zmiri-genmc-estimate`: This enables estimation of the concurrent execution space and verification time, before running the full verification. This should help users detect when their program is too complex to fully verify in a reasonable time. This will explore enough executions to make a good estimation, but at least 10 and at most `estimation-max` executions.
- `-Zmiri-genmc-estimation-max={MAX_ITERATIONS}`: Set the maximum number of executions that will be explored during estimation (default: 1000).
- `-Zmiri-genmc-print-exec-graphs={none,explored,blocked,all}`: Make GenMC print the execution graph of the program after every explored, every blocked, or after every execution (default: None).
- `-Zmiri-genmc-print-exec-graphs`: Shorthand for suffix `=explored`.
- `-Zmiri-genmc-print-genmc-output`: Print the output that GenMC provides. NOTE: this output is quite verbose and the events in the printed execution graph are hard to map back to the Rust code location they originate from.
- `-Zmiri-genmc-log=LOG_LEVEL`: Change the log level for GenMC. Default: `warning`.
  - `quiet`:    Disable logging.
  - `error`:    Print errors.
  - `warning`:  Print errors and warnings.
  - `tip`:      Print errors, warnings and tips.
  - If Miri is built with debug assertions, there are additional log levels available (downgraded to `tip` without debug assertions):
    - `debug1`:   Print revisits considered by GenMC.
    - `debug2`:   Print the execution graph after every memory access.
    - `debug3`:   Print reads-from values considered by GenMC.
- `-Zmiri-genmc-verbose`: Show more information, such as estimated number of executions, and time taken for verification.

#### Regular Miri parameters useful for GenMC mode

- `-Zmiri-disable-weak-memory-emulation`: Disable any weak memory effects (effectively upgrading all atomic orderings in the program to `SeqCst`). This option may reduce the number of explored program executions, but any bugs related to weak memory effects will be missed. This option can help determine if an error is caused by weak memory effects (i.e., if it disappears with this option enabled).

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

### Formatting the C++ code

For formatting the C++ code we provide a `.clang-format` file in the `genmc-sys` directory.
With `clang-format` installed, run this command to format the c++ files (replace the `-i` with `--dry-run` to just see the changes.):
```
find ./genmc-sys/cpp/ -name "*.cpp" -o -name "*.hpp" | xargs clang-format --style=file:"./genmc-sys/.clang-format" -i
```
NOTE: this is currently not done automatically on pull requests to Miri.

<!-- FIXME(genmc): explain how submitting code to GenMC should be handled. -->

<!-- FIXME(genmc): explain development. -->
