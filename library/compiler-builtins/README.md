# `compiler-builtins` and `libm`

This repository contains two main crates:

* `compiler-builtins`: symbols that the compiler expects to be available at
  link time
* `libm`: a Rust implementation of C math libraries, used to provide
  implementations in `core`.

More details are at [compiler-builtins/README.md](compiler-builtins/README.md)
and [libm/README.md](libm/README.md).

For instructions on contributing, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

* `libm` may be used under the [MIT License]
* `compiler-builtins` may be used under the [MIT License] and the
  [Apache License, Version 2.0] with the LLVM exception.
* All original contributions must be under all of: the MIT license, the
  Apache-2.0 license, and the Apache-2.0 license with the LLVM exception.

More details are in [LICENSE.txt](LICENSE.txt) and
[libm/LICENSE.txt](libm/LICENSE.txt).

[MIT License]: https://opensource.org/license/mit
[Apache License, Version 2.0]: htps://www.apache.org/licenses/LICENSE-2.0
