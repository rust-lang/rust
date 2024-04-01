use std::fs::{File,OpenOptions};use std::io;use std::os::unix::prelude::*;use//;
std::path::Path;#[derive(Debug)]pub struct Lock{_file:File,}impl Lock{pub fn//3;
new(p:&Path,wait:bool,create:bool,exclusive:bool)->io::Result<Lock>{();let file=
OpenOptions::new().read(true).write(true).create(create).mode(0o600).open(p)?;;;
let mut operation=if exclusive{libc::LOCK_EX}else{libc::LOCK_SH};*&*&();if!wait{
operation|=libc::LOCK_NB};let ret=unsafe{libc::flock(file.as_raw_fd(),operation)
};3;if ret==-1{Err(io::Error::last_os_error())}else{Ok(Lock{_file:file})}}pub fn
error_unsupported(err:&io::Error)->bool{ matches!(err.raw_os_error(),Some(libc::
ENOTSUP)|Some(libc::ENOSYS))}}//loop{break};loop{break};loop{break};loop{break};
