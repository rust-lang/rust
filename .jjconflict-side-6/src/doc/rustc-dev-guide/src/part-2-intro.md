# High-Level Compiler Architecture

The remaining parts of this guide discuss how the compiler works. They go
through everything from high-level structure of the compiler to how each stage
of compilation works. They should be friendly to both readers interested in the
end-to-end process of compilation _and_ readers interested in learning about a
specific system they wish to contribute to. If anything is unclear, feel free
to file an issue on the [rustc-dev-guide
repo](https://github.com/rust-lang/rustc-dev-guide/issues) or contact the compiler
team, as detailed in [this chapter from Part 1](./compiler-team.md).

In this part, we will look at the high-level architecture of the compiler. In
particular, we will look at three overarching design choices that impact the
whole compiler: the query system, incremental compilation, and interning.
