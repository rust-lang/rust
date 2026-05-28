# Autodiff Type-Trees Type Analysis Tests

This directory contains run-make tests for the autodiff type-trees type analysis functionality. These tests verify that the autodiff compiler correctly analyzes and tracks type information for different Rust types during automatic differentiation.

## What These Tests Do

Each test compiles a simple Rust function with the `#[autodiff_reverse]` attribute and verifies that the compiler:

1. **Correctly identifies type information** in the generated LLVM IR
2. **Tracks type annotations** for variables and operations
3. **Preserves type context** through the autodiff transformation process

The tests capture the stdout from the autodiff compiler (which contains type analysis information) and verify it matches expected patterns using FileCheck.
