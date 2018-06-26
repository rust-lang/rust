// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "futures_api",
            reason = "futures in libcore are unstable",
            issue = "50547")]

use fmt;
use mem;
use super::{TaskObj, LocalTaskObj};

/// Provides the reason that an executor was unable to spawn.
pub struct SpawnErrorKind {
    _hidden: (),
}

impl fmt::Debug for SpawnErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("SpawnErrorKind")
            .field(&"shutdown")
            .finish()
    }
}

impl SpawnErrorKind {
    /// Spawning is failing because the executor has been shut down.
    pub fn shutdown() -> SpawnErrorKind {
        SpawnErrorKind { _hidden: () }
    }

    /// Check whether this error is the `shutdown` error.
    pub fn is_shutdown(&self) -> bool {
        true
    }
}

/// The result of a failed spawn
#[derive(Debug)]
pub struct SpawnObjError {
    /// The kind of error
    pub kind: SpawnErrorKind,

    /// The task for which spawning was attempted
    pub task: TaskObj,
}

/// The result of a failed spawn
#[derive(Debug)]
pub struct SpawnLocalObjError {
    /// The kind of error
    pub kind: SpawnErrorKind,

    /// The task for which spawning was attempted
    pub task: LocalTaskObj,
}

impl SpawnLocalObjError {
    /// Converts the `SpawnLocalObjError` into a `SpawnObjError`
    /// To make this operation safe one has to ensure that the `UnsafeTask`
    /// instance from which the `LocalTaskObj` stored inside was created
    /// actually implements `Send`.
    pub unsafe fn as_spawn_obj_error(self) -> SpawnObjError {
        // Safety: Both structs have the same memory layout
        mem::transmute::<SpawnLocalObjError, SpawnObjError>(self)
    }
}

impl From<SpawnObjError> for SpawnLocalObjError {
    fn from(error: SpawnObjError) -> SpawnLocalObjError {
        unsafe {
            // Safety: Both structs have the same memory layout
            mem::transmute::<SpawnObjError, SpawnLocalObjError>(error)
        }
    }
}
