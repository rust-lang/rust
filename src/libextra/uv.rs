// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Rust bindings to libuv
 *
 * This is the base-module for various levels of bindings to
 * the libuv library.
 *
 * These modules are seeing heavy work, currently, and the final
 * API layout should not be inferred from its current form.
 *
 * This base module currently contains a historical, rust-based
 * implementation of a few libuv operations that hews closely to
 * the patterns of the libuv C-API. It was used, mostly, to explore
 * some implementation details and will most likely be deprecated
 * in the near future.
 *
 * The `ll` module contains low-level mappings for working directly
 * with the libuv C-API.
 *
 * The `hl` module contains a set of tools library developers can
 * use for interacting with an active libuv loop. This modules's
 * API is meant to be used to write high-level,
 * rust-idiomatic abstractions for utilizes libuv's asynchronous IO
 * facilities.
 */

pub use ll = super::uv_ll;
pub use iotask = uv_iotask;
pub use global_loop = uv_global_loop;
