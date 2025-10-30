# MIR Transform Passes

This crate implements optimization passes that transform MIR (Mid-level Intermediate Representation) to improve code quality and performance.

## Key Optimization Passes

### Destination Propagation (`dest_prop.rs`)
Eliminates redundant copy assignments like `dest = src` by unifying the storage of `dest` and `src`.
- **When to modify**: Adding new assignment patterns, improving liveness analysis
- **Key challenges**: Ensuring soundness with address-taken locals, handling storage lifetimes
- **Performance notes**: Past implementations had O(l² * s) complexity issues; current version uses conflict matrices

### Global Value Numbering (`gvn.rs`)
Detects and eliminates redundant computations by identifying values with the same symbolic representation.
- **When to modify**: Adding new value types, improving constant evaluation
- **Key challenges**: Handling non-deterministic constants, pointer provenance
- **Performance notes**: Largest pass (~2000 lines); careful about evaluation costs

### Dataflow Constant Propagation (`dataflow_const_prop.rs`)
Propagates scalar constants through the program using dataflow analysis.
- **When to modify**: Extending to non-scalar types, improving evaluation precision
- **Key challenges**: Place limits to avoid compile-time explosions
- **Performance notes**: Has BLOCK_LIMIT (100) and PLACE_LIMIT (100) guards

### Inlining (`inline.rs`, `inline/`)
Replaces function calls with the body of the called function.
- **When to modify**: Tuning heuristics, handling new calling conventions
- **Key challenges**: Avoiding cycles, cost estimation, handling generics
- **Performance notes**: Uses thresholds (30-100 depending on context)

## Adding a New Pass

1. Create your pass file in `src/`
2. Implement `MirPass<'tcx>` trait:
   - `is_enabled`: When the pass should run
   - `run_pass`: The transformation logic
   - `is_required`: Whether this is a required pass
3. Register in `lib.rs` within `mir_opts!` macro
4. Add to appropriate phase in `run_optimization_passes`
5. Add tests in `tests/mir-opt/`

## Testing

Run specific MIR opt tests:
```bash
./x.py test tests/mir-opt/dest_prop.rs
./x.py test tests/mir-opt/gvn.rs
./x.py test tests/mir-opt/dataflow-const-prop
./x.py test tests/mir-opt/inline
```

## Pass Ordering

Passes are organized into phases (see `lib.rs`):
- Early passes (cleanup, simplification)
- Analysis-driven optimizations (inlining, const prop, GVN)
- Late passes (final cleanup, code size reduction)

Order matters! For example:
- `SimplifyCfg` before `GVN` (cleaner CFG)
- `GVN` before `DeadStoreElimination` (more values identified)
- `SimplifyLocals` after most passes (remove unused locals)

## Known Limitations and Issues

### ConstParamHasTy and Drop Shim Builder (#127030)
Drop glue generation for types with const parameters has a param-env construction issue:
- **Problem**: `build_drop_shim` (in `shim.rs`) constructs its typing environment using the `drop_in_place` intrinsic's DefId, not the dropped type's DefId
- **Impact**: The param-env lacks `ConstArgHasType` predicates, causing panics when MIR generation needs const param types
- **Workaround**: Inlining of drop glue is disabled for types containing const params until they're fully monomorphized (see `inline.rs:746`)
- **Proper fix**: Requires synthesizing a typing environment that merges predicates from both drop_in_place and the dropped type

This affects types like `struct Foo<const N: usize> { ... }` with Drop implementations.

## Common Patterns

### Visiting MIR
- Use `rustc_middle::mir::visit::Visitor` for read-only traversal
- Use `MutVisitor` for in-place modifications
- Call `visit_body_preserves_cfg` to keep the CFG structure

### Creating new values
```rust
let ty = Ty::new_tuple(tcx, &[tcx.types.i32, tcx.types.bool]);
let rvalue = Rvalue::Aggregate(box AggregateKind::Tuple, vec![op1, op2]);
```

### Cost checking
Use `CostChecker` (from `cost_checker.rs`) to estimate the cost of inlining or other transformations.

## Performance Considerations

- **Compile time vs runtime**: These passes increase compile time to reduce runtime
- **Limits**: Many passes have size/complexity limits to prevent exponential blowup
- **Profiling**: Use `-Ztime-passes` to see pass timings
- **Benchmarking**: Run `./x.py bench` with rustc-perf suite

## References

- [MIR documentation](https://rustc-dev-guide.rust-lang.org/mir/)
- [Optimization passes](https://rustc-dev-guide.rust-lang.org/mir/optimizations.html)
- [Dataflow framework](https://rustc-dev-guide.rust-lang.org/mir/dataflow.html)
