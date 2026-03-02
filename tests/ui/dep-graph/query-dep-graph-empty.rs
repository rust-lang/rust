//@ build-pass
//@ compile-flags: -Zquery-dep-graph --crate-type lib
//@ edition: 2021

// This file is intentionally left empty to reproduce issue #153199.
// rustc used to ICE when generating a dependency graph for an empty file
// because early queries would panic when unwrapping an uninitialized graph.
