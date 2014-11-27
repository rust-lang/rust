// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use self::Req::*;

use libc::{mod, pid_t, c_void, c_int};
use c_str::CString;
use io::{mod, IoResult, IoError};
use mem;
use os;
use ptr;
use prelude::*;
use io::process::{ProcessExit, ExitStatus, ExitSignal};
use collections;
use path::BytesContainer;
use hash::Hash;

use sys::{mod, retry, c, wouldblock, set_nonblocking, ms_to_timeval};
use sys::fs::FileDesc;
use sys_common::helper_thread::Helper;
use sys_common::{AsFileDesc, mkerr_libc, timeout};

pub use sys_common::ProcessConfig;

helper_init!(static HELPER: Helper<Req>)

/// The unique id of the process (this should never be negative).
pub struct Process {
    pub pid: pid_t
}

enum Req {
    NewChild(libc::pid_t, Sender<ProcessExit>, u64),
}

impl Process {
    pub fn id(&self) -> pid_t { unimplemented!() }

    pub unsafe fn kill(&self, signal: int) -> IoResult<()> { unimplemented!() }

    pub unsafe fn killpid(pid: pid_t, signal: int) -> IoResult<()> { unimplemented!() }

    pub fn spawn<K, V, C, P>(cfg: &C, in_fd: Option<P>,
                              out_fd: Option<P>, err_fd: Option<P>)
                              -> IoResult<Process>
        where C: ProcessConfig<K, V>, P: AsFileDesc,
              K: BytesContainer + Eq + Hash, V: BytesContainer
    { unimplemented!() }

    pub fn wait(&self, deadline: u64) -> IoResult<ProcessExit> { unimplemented!() }

    pub fn try_wait(&self) -> Option<ProcessExit> { unimplemented!() }
}

fn with_argv<T>(prog: &CString, args: &[CString],
                cb: proc(*const *const libc::c_char) -> T) -> T { unimplemented!() }

fn with_envp<K, V, T>(env: Option<&collections::HashMap<K, V>>,
                      cb: proc(*const c_void) -> T) -> T
    where K: BytesContainer + Eq + Hash, V: BytesContainer
{ unimplemented!() }

fn translate_status(status: c_int) -> ProcessExit { unimplemented!() }
