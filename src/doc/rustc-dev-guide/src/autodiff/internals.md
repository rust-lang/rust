The `std::autodiff` module in Rust allows differentiable programming:

```rust
#![feature(autodiff)]
use std::autodiff::*;

// f(x) = x * x, f'(x) = 2.0 * x
// bar therefore returns (x * x, 2.0 * x)
#[autodiff_reverse(bar, Active, Active)]
fn foo(x: f32) -> f32 { x * x }

fn main() {
    assert_eq!(bar(3.0, 1.0), (9.0, 6.0));
    assert_eq!(bar(4.0, 1.0), (16.0, 8.0));
}
```

The detailed documentation for the `std::autodiff` module is available at [std::autodiff](https://doc.rust-lang.org/nightly/std/autodiff/index.html).

Differentiable programming is used in various fields like numerical computing, [solid mechanics][ratel], [computational chemistry][molpipx], [fluid dynamics][waterlily] or for Neural Network training via Backpropagation, [ODE solver][diffsol], [differentiable rendering][libigl], [quantum computing][catalyst], and climate simulations.

[ratel]: https://gitlab.com/micromorph/ratel
[molpipx]: https://arxiv.org/abs/2411.17011
[waterlily]: https://github.com/WaterLily-jl/WaterLily.jl
[diffsol]: https://github.com/martinjrobins/diffsol
[libigl]: https://github.com/alecjacobson/libigl-enzyme-example?tab=readme-ov-file#run
[catalyst]: https://github.com/PennyLaneAI/catalyst


`std::autodiff` is currently based on Enzyme, an LLVM based tool for automatic differentation.
There are three main reasons for relying on compiler based autodiff:

- **Usability**: Current autodiff crates do not support normal Rust programs. They either enforce a custom DSL, require the usage of library provided types (instead of e.g. slices or arrays), or are limited to scalar functions. Compiler based autodiff allows users to write normal Rust code, including arrays, slices, user-defined structs and enums, control flow, and more.
- **Performance**: Most existing Rust autodiff approaches have a constant overhead per operation.
  This can easily be amortized for ML applications which have few expensive operations on large tensors.
  It is, however, often unacceptable for applications in the HPC or scientific computing field.
  By working on (optimized) LLVM IR, compiler based autodiff can achieve [significantly][Enzyme] better performance in those cases.
- **Features**: By operating on such a low level and sharing the implementation with other LLVM based languages, we can leverage the large amount of work already done in the Enzyme project.
  For example, we can support Rust code calling MPI routines, or GPU code, including libraries like CuBLAS.

[Enzyme]: https://dl.acm.org/doi/10.5555/3495724.3496770
