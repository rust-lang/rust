# Current limitations
 
## Safety and Soundness

Enzyme currently assumes that the user passes shadow arguments (`dx`, `dy`, ...) of appropriate size. Under Reverse Mode, we additionally assume that shadow arguments are mutable. In Reverse Mode we adjust the outermost pointer or reference to be mutable. Therefore `&f32` will receive the shadow type `&mut f32`. However, we do not check length for other types than slices (e.g. enums, Vec). We also do not enforce mutability of inner references, but will warn if we recognize them. We do intend to add additional checks over time.

## ABI adjustments

In some cases, a function parameter might get lowered in a way that we currently don't handle correctly, leading to a compile time type mismatch in the `rustc_codegen_llvm` backend. Here are some [examples](https://github.com/EnzymeAD/rust/issues/105).

## Compile Times

Enzyme will often achieve excellent runtime performance, but might increase your compile time by a large factor. For Rust, we already have made significant improvements and have a list of further improvements planed - please reach out if you have time to help here.

### Type Analysis

Most of the times, Type Analysis (TA) is the reason of large (>5x) compile time increases when using Enzyme. This poster explains why we need to run Type Analysis in the bottom left part: [Poster Link](https://c.wsmoses.com/posters/Enzyme-llvmdev.pdf).

We intend to increase the number of locations where we pass down Type information based on Rust types, which in turn will reduce the number of locations where Enzyme has to run Type Analysis, which will help compile times.

### Duplicated Optimizations

The key reason for Enzyme offering often excellent performance is that Enzyme differentiates already optimized LLVM-IR. However, we also (have to) run LLVM's optimization pipeline after differentiating, to make sure that the code which Enzyme generates is optimized properly. As a result you should have excellent runtime performance (please fill an issue if not), but at a compile time cost for running optimizations twice.

### Fat-LTO 

The usage of `#[autodiff(...)]` currently requires compiling your project with Fat-LTO. We technically only need LTO if the function being differentiated calls functions in other compilation units. Therefore, other solutions are possible, but this is the most simple one to get started. 
