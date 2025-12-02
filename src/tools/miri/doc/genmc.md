# **(WIP)** Documentation for Miri-GenMC

**NOTE: GenMC mode is not yet fully implemented, and has [several correctness issues](https://github.com/rust-lang/miri/issues/4572). Using GenMC mode currently requires manually compiling Miri, see [Usage](#usage).**


[GenMC](https://github.com/MPI-SWS/genmc) is a stateless model checker for exploring concurrent executions of a program.
Miri-GenMC integrates that model checker into Miri.

Miri in GenMC mode takes a program as input like regular Miri, but instead of running it once, the program is executed repeatedly, until all possible executions allowed by the Rust memory model are explored.
This includes all possible thread interleavings and all allowed return values for atomic operations, including cases that are very rare to encounter on actual hardware.
(However, this does not include other sources of non-determinism, such as the absolute addresses of allocations.
It is hence still possible to have latent bugs in a test case even if they passed GenMC.)

GenMC requires the input program to be bounded, i.e., have finitely many possible executions, otherwise it will not terminate.
Any loops that may run infinitely must be replaced or bounded (see below).

GenMC makes use of Dynamic Partial Order Reduction (DPOR) to reduce the number of executions that must be explored, but the runtime can still be super-exponential in the size of the input program (number of threads and amount of interaction between threads).
Large programs may not be verifiable in a reasonable amount of time.

## Usage

For testing/developing Miri-GenMC:
- install all [dependencies required by GenMC](https://github.com/MPI-SWS/genmc?tab=readme-ov-file#dependencies)
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

### Eliminating unbounded loops

As mentioned above, GenMC requires all loops to be bounded.
Otherwise, it is not possible to exhaustively explore all executions.
Currently, Miri-GenMC has no support for automatically bounding loops, so this needs to be done manually.

#### Bounding loops without side effects

The easiest case is that of a loop that simply spins until it observes a certain condition, without any side effects.
Such loops can be limited to one iteration, as demonstrated by the following example:

```rust
#[cfg(miri)]
unsafe extern "Rust" {
  // This is a special function that Miri provides.
  // It blocks the thread calling this function if the condition is false.
  pub unsafe fn miri_genmc_assume(condition: bool);
}

// This functions loads an atomic boolean in a loop until it is true.
// GenMC will explore all executions where this does 1, 2, ..., ∞ loads, which means the verification will never terminate.
fn spin_until_true(flag: &AtomicBool) {
  while !flag.load(Relaxed) {
    std::hint::spin_loop();
  }
}

// By replacing this loop with an assume statement, the only executions that will be explored are those with exactly 1 load that observes the expected value.
// Incorrect use of assume statements can lead GenMC to miss important executions, so it is marked `unsafe`.
fn spin_until_true_genmc(flag: &AtomicBool) {
  unsafe { miri_genmc_assume(flag.load(Relaxed)) };
}
```

#### Bounding loops with side effects

Some loops do contain side effects, meaning the number of explored iterations affects the rest of the program.
Replacing the loop with one iteration like we did above would mean we miss all those possible executions.

In such a case, the loop can be limited to a fixed number of iterations instead.
The choice of iteration limit trades off verification time for possibly missing bugs requiring more iterations.

```rust
/// The loop in this function has a side effect, which is to increment the counter for the number of iterations.
/// Instead of replacing the loop entirely (which would miss all executions with `count > 0`), we limit the loop to at most 3 iterations.
fn count_until_true_genmc(flag: &AtomicBool) -> u64 {
  let mut count = 0;
  while !flag.load(Relaxed) {
    count += 1;
    std::hint::spin_loop();
    // Any execution that takes more than 3 iterations will not be explored.
    unsafe { miri_genmc_assume(count <= 3) };
  }
  count
}
```

<!-- FIXME: update the code above once Miri supports a loop bounding features like GenMC's `--unroll=N`. -->
<!-- FIXME: update this section once Miri-GenMC supports automatic program transformations (like spinloop-assume replacement). -->

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
