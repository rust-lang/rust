// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate addr2line;
extern crate gimli;
extern crate object;

use libc::{mmap, size_t, PROT_READ, MAP_SHARED, MAP_FAILED};

use io;
use ptr;

use cell::UnsafeCell;
use marker::Sync;
use sys::backtrace::BacktraceContext;
use sys_common::backtrace::Frame;
use os::unix::io::AsRawFd;
use fs::File;
use slice::from_raw_parts;
use self::addr2line::Context;
use self::gimli::{EndianRcSlice, RunTimeEndian};

struct ThreadSafe<T>(UnsafeCell<T>);

unsafe impl<T> Sync for ThreadSafe<T> {}

macro_rules! err {
    ($e:expr) => {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            $e)
        );
    }
}

struct Ctx {
    ctx: Context<EndianRcSlice<RunTimeEndian>>,
    _file: File,
}

impl Ctx {
    fn init() -> io::Result<Ctx> {
        let (_filename, file) = ::sys::backtrace::gnu::get_executable_filename()?;
        let file_len = file.metadata()?.len();

        let map_ptr = unsafe {
            mmap(ptr::null_mut(),
                file_len as size_t,
                PROT_READ,
                MAP_SHARED,
                file.as_raw_fd(),
                0)
        };
        if map_ptr == MAP_FAILED {
            err!("mapping the executable into memory failed");
        }

        let map = unsafe { from_raw_parts(map_ptr as * mut u8, file_len as usize) };
        let object_file = match object::File::parse(&*map) {
            Ok(v) => v,
            Err(_) => err!("could not parse the object file for backtrace creation"),
        };
        let ctx = match Context::new(&object_file) {
            Ok(v) => v,
            Err(_) => err!("could not create backtrace parsing context"),
        };

        Ok(Ctx {
            ctx,
            _file: file,
        })
    }
    fn with_static<T, F: FnOnce(&Ctx) -> io::Result<T>>(f: F) -> io::Result<T> {
        static CTX_CELL: ThreadSafe<Option<Ctx>> = ThreadSafe(UnsafeCell::new(None));
        let cell_ref: &mut Option<_> = unsafe { &mut *CTX_CELL.0.get() };
        if cell_ref.is_none() {
            *cell_ref = Some(Ctx::init()?);
        }
        let ctx = if let Some(c) = cell_ref {
            c
        } else {
            unreachable!()
        };
        f(ctx)
    }
}

pub fn foreach_symbol_fileline<F>(frame: Frame,
                                  mut f: F,
                                  _: &BacktraceContext) -> io::Result<bool>
where F: FnMut(&[u8], u32) -> io::Result<()>
{
    Ctx::with_static(|ctx|{
        let mut frames_iter = match ctx.ctx.find_frames(frame.exact_position as u64) {
            Ok(v) => v,
            Err(_) => err!("error during binary parsing"),
        };
        for frame_opt in frames_iter.next() {
            let loc_opt = frame_opt.and_then(|v| v.location);
            let file_line_opt = loc_opt.map(|v| (v.file, v.line));
            if let Some((Some(file), Some(line))) = file_line_opt {
                f(file.as_bytes(), line as u32)?;
            }
        }
        // FIXME: when should we return true here?
        Ok(false)
    })
}

/// Converts a pointer to symbol to its string value.
pub fn resolve_symname<F>(frame: Frame,
                          callback: F,
                          _: &BacktraceContext) -> io::Result<()>
    where F: FnOnce(Option<&str>) -> io::Result<()>
{
    Ctx::with_static(|ctx|{
        let mut frames_iter = match ctx.ctx.find_frames(frame.symbol_addr as u64) {
            Ok(v) => v,
            Err(_) => err!("error during binary parsing"),
        };
        let frame_opt = match frames_iter.next() {
            Ok(v) => v,
            Err(_) => err!("error during symbolification"),
        };
        match frame_opt.and_then(|v| v.function) {
            Some(n) => match n.raw_name() {
                Ok(v) => callback(Some(&*v))?,
                Err(_) => err!("error during name resolval"),
            },
            None => callback(None)?,
        }
        Ok(())
    })
}
