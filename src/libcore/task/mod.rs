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

//! Types and Traits for working with asynchronous tasks.

mod context;
pub use self::context::Context;

mod executor;
pub use self::executor::Executor;

mod poll;
pub use self::poll::Poll;

mod spawn_error;
pub use self::spawn_error::{SpawnErrorKind, SpawnObjError};

mod task;
pub use self::task::{TaskObj, UnsafeTask};

mod wake;
pub use self::wake::{Waker, LocalWaker, UnsafeWake};
