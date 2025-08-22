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

The detailed documentation for the `std::autodiff` module is available at [std::autodiff](https://doc.rust-lang.org/std/autodiff/index.html).

Differentiable programming is used in various fields like numerical computing, [solid mechanics][ratel], [computational chemistry][molpipx], [fluid dynamics][waterlily] or for Neural Network training via Backpropagation, [ODE solver][diffsol], [differentiable rendering][libigl], [quantum computing][catalyst], and climate simulations.

[ratel]: https://gitlab.com/micromorph/ratel
[molpipx]: https://arxiv.org/abs/2411.17011v
[waterlily]: https://github.com/WaterLily-jl/WaterLily.jl
[diffsol]: https://github.com/martinjrobins/diffsol
[libigl]: https://github.com/alecjacobson/libigl-enzyme-example?tab=readme-ov-file#run
[catalyst]: https://github.com/PennyLaneAI/catalyst
