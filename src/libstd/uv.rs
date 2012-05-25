#[doc = "
Rust bindings to libuv

This is the base-module for various levels of bindings to
the libuv library.

These modules are seeing heavy work, currently, and the final
API layout should not be inferred from its current form.

This base module currently contains a historical, rust-based
implementation of a few libuv operations that hews closely to
the patterns of the libuv C-API. It was used, mostly, to explore
some implementation details and will most likely be deprecated
in the near future.

The `ll` module contains low-level mappings for working directly
with the libuv C-API.

The `hl` module contains a set of tools library developers can
use for interacting with an active libuv loop. This modules's
API is meant to be used to write high-level,
rust-idiomatic abstractions for utilizes libuv's asynchronous IO
facilities.
"];

import ll = uv_ll;
export ll;

import iotask = uv_iotask;
export iotask;

import global_loop = uv_global_loop;
export global_loop;
