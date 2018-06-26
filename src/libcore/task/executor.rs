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
use super::{TaskObj, LocalTaskObj};

/// A task executor.
///
/// A *task* is a `()`-producing async value that runs at the top level, and will
/// be `poll`ed until completion. It's also the unit at which wake-up
/// notifications occur. Executors, such as thread pools, allow tasks to be
/// spawned and are responsible for putting tasks onto ready queues when
/// they are woken up, and polling them when they are ready.
pub trait Executor {
    /// Spawn the given task, polling it until completion.
    ///
    /// # Errors
    ///
    /// The executor may be unable to spawn tasks, either because it has
    /// been shut down or is resource-constrained.
    fn spawn_obj(&mut self, task: TaskObj) -> Result<(), SpawnObjError>;

    /// Determine whether the executor is able to spawn new tasks.
    ///
    /// # Returns
    ///
    /// An `Ok` return means the executor is *likely* (but not guaranteed)
    /// to accept a subsequent spawn attempt. Likewise, an `Err` return
    /// means that `spawn` is likely, but not guaranteed, to yield an error.
    #[inline]
    fn status(&self) -> Result<(), SpawnErrorKind> {
        Ok(())
    }
}

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
