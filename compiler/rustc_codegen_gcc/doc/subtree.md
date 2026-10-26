# Subtree sync procedure

`rustc_codegen_gcc` is using [Josh] to perform subtree synces with the `rust-lang/rust` repository.

[Josh]: https://josh-project.dev/docs/intro.html

The procedure for performing subtree synchronizations is documented [here][rdg-josh].

[rdg-josh]: https://rustc-dev-guide.rust-lang.org/external-repos.html#synchronizing-a-josh-subtree

## Pushing changes to `rust-lang/rust`

If the used GCC commit (recorded in `libgccjit.version`) changed in `rustc_codegen_gcc`, then you will have to update the `gcc` submodule in the `rust-lang/rust` repository in the push sync PR. The CI in `rust-lang/rust` should ensure that these two commit references match. You can simply update the `gcc` submodule commit and push it to your push sync PR once it has been created.
