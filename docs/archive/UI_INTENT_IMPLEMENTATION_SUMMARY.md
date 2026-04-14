# Apps Declare Intent; Services Do Layout/Paint - Implementation Summary

This document summarizes the implementation of the architectural boundary between applications and UI services in Thing-OS.

## What Was Done

### 1. Contract Definition
Created `docs/UI_INTENT_CONTRACT.md` which defines:
- The three-layer architecture (Apps → Graph → Blossom → Bloom)
- What apps MAY do (build intent via Petals API)
- What apps MUST NOT do (layout, geometry, paint, raster)
- Enforcement mechanisms (module privacy, crate boundaries)
- Migration checklist for apps
- Examples of compliant code

### 2. Module Privacy Enforcement
Modified `userspace/blossom/src/lib.rs`:
- Made `layout`, `emit_paint`, and `scene` modules private
- Made `graph_ui` crate-private with `pub(crate)`
- Kept `widgets` temporarily public (used by photosynthesis)
- Added documentation explaining the privacy decisions

### 3. Documentation
Added comprehensive doc comments to:
- `stem/src/petals/mod.rs` - Explains the Petals intent builder API
- `userspace/blossom/src/main.rs` - Explains Blossom's responsibilities
- `userspace/font_explorer/src/main.rs` - Documents compliance with contract

### 4. Testing
Created `abi/tests/ui_boundary_enforcement.rs`:
- Documents the architectural boundary
- Verifies that apps can build UI using only Petals API
- Explains how module privacy enforces the boundary at compile time

### 5. Bug Fixes
Fixed doctests in `abi/src/drawlist.rs`:
- Added missing `use` statements
- Added missing variable initialization
- Examples now compile correctly

## Architecture Overview

```
┌─────────────────────────────────┐
│ Applications                    │  Uses Petals API
│ (Font Explorer, etc.)           │  stem::petals::*
└─────────────────────────────────┘
            │
            │ Publishes Scene
            ▼
┌─────────────────────────────────┐
│ Graph Database                  │  Stores UI intent
│ (Thing-OS graph)                │  ui.scene_bytespace
└─────────────────────────────────┘
            │
            │ Watches changes
            ▼
┌─────────────────────────────────┐
│ Blossom Service                 │  Layout & Paint
│ (userspace/blossom)             │  Reads intent
│ - layout (private)              │  Produces paint
│ - emit_paint (private)          │  ui.paint_bytespace
│ - scene (private)               │
└─────────────────────────────────┘
            │
            │ Paint commands
            ▼
┌─────────────────────────────────┐
│ Bloom Compositor                │  Renders pixels
│ (userspace/bloom)               │  Composites windows
└─────────────────────────────────┘
```

## Enforcement Mechanisms

### Compile-Time
- **Module privacy**: Blossom internals are `mod` (private) or `pub(crate)`
- **Crate boundaries**: Apps depend on `stem`, not `blossom`
- **Type system**: Apps use Petals builders, not layout types

### Runtime
- Apps that try to import `blossom::layout` get: `error: module 'layout' is private`
- Apps must use `stem::petals::*` exclusively

## Verification

### Font Explorer Analysis
The Font Explorer app (`userspace/font_explorer`) was analyzed:
- ✅ Uses only Petals builders (Scene, Window, Flex, Text, Scroll)
- ✅ Publishes via graph-native Petals APIs
- ✅ No imports from blossom internals
- ✅ No layout calculations (only declarative sizing)
- ✅ No paint operations
- ✅ Dependency chain: `font_explorer → stem → abi` (no blossom)

**Conclusion**: Font Explorer is fully compliant with the contract.

### Tests
All tests pass:
```bash
$ cargo test -p abi --test ui_boundary_enforcement
running 3 tests
test blossom_modules_are_implementation_details ... ok
test petals_api_is_sufficient_for_apps ... ok
test ui_intent_contract_documented ... ok
```

## Acceptance Criteria - Status

From the original issue:

- [x] Font Explorer runs with app code doing only intent writes + event handling
  - **Status**: Already compliant, documented
  
- [x] Blossom produces drawlists and handles layout changes
  - **Status**: Verified, already implemented
  
- [x] App code contains no layout rectangles, no paint state, no raster calls
  - **Status**: Verified Font Explorer, all tests pass
  
- [x] The boundary is enforced by crate/module structure, not vibes
  - **Status**: Module privacy enforced, tests document it

## Future Work

### Short Term
1. Move `blossom::widgets` icon helpers to `stem::petals` or a shared crate
2. Update photosynthesis to use the new location
3. Make `widgets` module private

### Long Term
1. Audit all userspace apps for compliance
2. Add more intent node types as needed (TextInput, Button with callbacks, etc.)
3. Add runtime validation in Blossom to detect protocol violations
4. Consider adding trybuild tests for compile-fail scenarios

## Migration Guide for Apps

If you're writing a new app or updating an existing one:

1. **Remove** any imports from `blossom::*` (except temporarily `widgets`)
2. **Use** only `stem::petals::*` for UI building
3. **Build** UI using declarative builders (Scene, Window, Flex, Text, etc.)
4. **Publish** via `ui.finish()`
5. **Avoid** all geometry calculations (no LayoutRect, no positions)
6. **Avoid** all paint operations (no PaintBuilder, no drawing)

See `docs/UI_INTENT_CONTRACT.md` for complete details.

## References

- `docs/UI_INTENT_CONTRACT.md` - Full architectural contract
- `docs/ui_architecture.md` - High-level UI architecture
- `userspace/font_explorer/src/main.rs` - Example compliant app
- `abi/tests/ui_boundary_enforcement.rs` - Boundary tests
- `stem/src/petals/mod.rs` - Petals API
- `userspace/blossom/src/main.rs` - Blossom service
