// FIXME: This is a complete copy of `cargo/src/cargo/util/read2.rs`
// Consider unify the read2() in libstd, cargo and this to prevent further code duplication.

#[cfg(test)]
mod tests;

pub use self::imp::read2;
use std::io::{self, Write};
use std::mem::replace;
use std::process::{Child, Output};

pub fn read2_abbreviated(mut child: Child, filter_paths_from_len: &[String]) -> io::Result<Output> {
    let mut stdout = ProcOutput::new();
    let mut stderr = ProcOutput::new();

    drop(child.stdin.take());
    read2(
        child.stdout.take().unwrap(),
        child.stderr.take().unwrap(),
        &mut |is_stdout, data, _| {
            if is_stdout { &mut stdout } else { &mut stderr }.extend(data, filter_paths_from_len);
            data.clear();
        },
    )?;
    let status = child.wait()?;

    Ok(Output { status, stdout: stdout.into_bytes(), stderr: stderr.into_bytes() })
}

const HEAD_LEN: usize = 160 * 1024;
const TAIL_LEN: usize = 256 * 1024;

// Whenever a path is filtered when counting the length of the output, we need to add some
// placeholder length to ensure a compiler emitting only filtered paths doesn't cause a OOM.
//
// 32 was chosen semi-arbitrarily: it was the highest power of two that still allowed the test
// suite to pass at the moment of implementing path filtering.
const FILTERED_PATHS_PLACEHOLDER_LEN: usize = 32;

enum ProcOutput {
    Full { bytes: Vec<u8>, filtered_len: usize },
    Abbreviated { head: Vec<u8>, skipped: usize, tail: Box<[u8]> },
}

impl ProcOutput {
    fn new() -> Self {
        ProcOutput::Full { bytes: Vec::new(), filtered_len: 0 }
    }

    fn extend(&mut self, data: &[u8], filter_paths_from_len: &[String]) {
        let new_self = match *self {
            ProcOutput::Full { ref mut bytes, ref mut filtered_len } => {
                let old_len = bytes.len();
                bytes.extend_from_slice(data);
                *filtered_len += data.len();

                // We had problems in the past with tests failing only in some environments,
                // due to the length of the base path pushing the output size over the limit.
                //
                // To make those failures deterministic across all environments we ignore known
                // paths when calculating the string length, while still including the full
                // path in the output. This could result in some output being larger than the
                // threshold, but it's better than having nondeterministic failures.
                //
                // The compiler emitting only excluded strings is addressed by adding a
                // placeholder size for each excluded segment, which will eventually reach
                // the configured threshold.
                for path in filter_paths_from_len {
                    let path_bytes = path.as_bytes();
                    // We start matching `path_bytes - 1` into the previously loaded data,
                    // to account for the fact a path_bytes might be included across multiple
                    // `extend` calls. Starting from `- 1` avoids double-counting paths.
                    let matches = (&bytes[(old_len.saturating_sub(path_bytes.len() - 1))..])
                        .windows(path_bytes.len())
                        .filter(|window| window == &path_bytes)
                        .count();
                    *filtered_len -= matches * path_bytes.len();

                    // We can't just remove the length of the filtered path from the output lenght,
                    // otherwise a compiler emitting only filtered paths would OOM compiletest. Add
                    // a fixed placeholder length for each path to prevent that.
                    *filtered_len += matches * FILTERED_PATHS_PLACEHOLDER_LEN;
                }

                let new_len = bytes.len();
                if (*filtered_len).min(new_len) <= HEAD_LEN + TAIL_LEN {
                    return;
                }

                let mut head = replace(bytes, Vec::new());
                let mut middle = head.split_off(HEAD_LEN);
                let tail = middle.split_off(middle.len() - TAIL_LEN).into_boxed_slice();
                let skipped = new_len - HEAD_LEN - TAIL_LEN;
                ProcOutput::Abbreviated { head, skipped, tail }
            }
            ProcOutput::Abbreviated { ref mut skipped, ref mut tail, .. } => {
                *skipped += data.len();
                if data.len() <= TAIL_LEN {
                    tail[..data.len()].copy_from_slice(data);
                    tail.rotate_left(data.len());
                } else {
                    tail.copy_from_slice(&data[(data.len() - TAIL_LEN)..]);
                }
                return;
            }
        };
        *self = new_self;
    }

    fn into_bytes(self) -> Vec<u8> {
        match self {
            ProcOutput::Full { bytes, .. } => bytes,
            ProcOutput::Abbreviated { mut head, mut skipped, tail } => {
                let mut tail = &*tail;

                // Skip over '{' at the start of the tail, so we don't later wrongfully consider this as json.
                // See <https://rust-lang.zulipchat.com/#narrow/stream/182449-t-compiler.2Fhelp/topic/Weird.20CI.20failure/near/321797811>
                while tail.get(0) == Some(&b'{') {
                    tail = &tail[1..];
                    skipped += 1;
                }

                write!(&mut head, "\n\n<<<<<< SKIPPED {} BYTES >>>>>>\n\n", skipped).unwrap();
                head.extend_from_slice(tail);
                head
            }
        }
    }
}

#[cfg(not(any(unix, windows)))]
mod imp {
    use std::io::{self, Read};
    use std::process::{ChildStderr, ChildStdout};

    pub fn read2(
        out_pipe: ChildStdout,
        err_pipe: ChildStderr,
        data: &mut dyn FnMut(bool, &mut Vec<u8>, bool),
    ) -> io::Result<()> {
        let mut buffer = Vec::new();
        out_pipe.read_to_end(&mut buffer)?;
        data(true, &mut buffer, true);
        buffer.clear();
        err_pipe.read_to_end(&mut buffer)?;
        data(false, &mut buffer, true);
        Ok(())
    }
}

#[cfg(unix)]
mod imp {
    use std::io;
    use std::io::prelude::*;
    use std::mem;
    use std::os::unix::prelude::*;
    use std::process::{ChildStderr, ChildStdout};

    pub fn read2(
        mut out_pipe: ChildStdout,
        mut err_pipe: ChildStderr,
        data: &mut dyn FnMut(bool, &mut Vec<u8>, bool),
    ) -> io::Result<()> {
        unsafe {
            libc::fcntl(out_pipe.as_raw_fd(), libc::F_SETFL, libc::O_NONBLOCK);
            libc::fcntl(err_pipe.as_raw_fd(), libc::F_SETFL, libc::O_NONBLOCK);
        }

        let mut out_done = false;
        let mut err_done = false;
        let mut out = Vec::new();
        let mut err = Vec::new();

        let mut fds: [libc::pollfd; 2] = unsafe { mem::zeroed() };
        fds[0].fd = out_pipe.as_raw_fd();
        fds[0].events = libc::POLLIN;
        fds[1].fd = err_pipe.as_raw_fd();
        fds[1].events = libc::POLLIN;
        let mut nfds = 2;
        let mut errfd = 1;

        while nfds > 0 {
            // wait for either pipe to become readable using `select`
            let r = unsafe { libc::poll(fds.as_mut_ptr(), nfds, -1) };
            if r == -1 {
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(err);
            }

            // Read as much as we can from each pipe, ignoring EWOULDBLOCK or
            // EAGAIN. If we hit EOF, then this will happen because the underlying
            // reader will return Ok(0), in which case we'll see `Ok` ourselves. In
            // this case we flip the other fd back into blocking mode and read
            // whatever's leftover on that file descriptor.
            let handle = |res: io::Result<_>| match res {
                Ok(_) => Ok(true),
                Err(e) => {
                    if e.kind() == io::ErrorKind::WouldBlock {
                        Ok(false)
                    } else {
                        Err(e)
                    }
                }
            };
            if !err_done && fds[errfd].revents != 0 && handle(err_pipe.read_to_end(&mut err))? {
                err_done = true;
                nfds -= 1;
            }
            data(false, &mut err, err_done);
            if !out_done && fds[0].revents != 0 && handle(out_pipe.read_to_end(&mut out))? {
                out_done = true;
                fds[0].fd = err_pipe.as_raw_fd();
                errfd = 0;
                nfds -= 1;
            }
            data(true, &mut out, out_done);
        }
        Ok(())
    }
}

#[cfg(windows)]
mod imp {
    use std::io;
    use std::os::windows::prelude::*;
    use std::process::{ChildStderr, ChildStdout};
    use std::slice;

    use miow::iocp::{CompletionPort, CompletionStatus};
    use miow::pipe::NamedPipe;
    use miow::Overlapped;
    use windows::Win32::Foundation::ERROR_BROKEN_PIPE;

    struct Pipe<'a> {
        dst: &'a mut Vec<u8>,
        overlapped: Overlapped,
        pipe: NamedPipe,
        done: bool,
    }

    pub fn read2(
        out_pipe: ChildStdout,
        err_pipe: ChildStderr,
        data: &mut dyn FnMut(bool, &mut Vec<u8>, bool),
    ) -> io::Result<()> {
        let mut out = Vec::new();
        let mut err = Vec::new();

        let port = CompletionPort::new(1)?;
        port.add_handle(0, &out_pipe)?;
        port.add_handle(1, &err_pipe)?;

        unsafe {
            let mut out_pipe = Pipe::new(out_pipe, &mut out);
            let mut err_pipe = Pipe::new(err_pipe, &mut err);

            out_pipe.read()?;
            err_pipe.read()?;

            let mut status = [CompletionStatus::zero(), CompletionStatus::zero()];

            while !out_pipe.done || !err_pipe.done {
                for status in port.get_many(&mut status, None)? {
                    if status.token() == 0 {
                        out_pipe.complete(status);
                        data(true, out_pipe.dst, out_pipe.done);
                        out_pipe.read()?;
                    } else {
                        err_pipe.complete(status);
                        data(false, err_pipe.dst, err_pipe.done);
                        err_pipe.read()?;
                    }
                }
            }

            Ok(())
        }
    }

    impl<'a> Pipe<'a> {
        unsafe fn new<P: IntoRawHandle>(p: P, dst: &'a mut Vec<u8>) -> Pipe<'a> {
            Pipe {
                dst: dst,
                pipe: NamedPipe::from_raw_handle(p.into_raw_handle()),
                overlapped: Overlapped::zero(),
                done: false,
            }
        }

        unsafe fn read(&mut self) -> io::Result<()> {
            let dst = slice_to_end(self.dst);
            match self.pipe.read_overlapped(dst, self.overlapped.raw()) {
                Ok(_) => Ok(()),
                Err(e) => {
                    if e.raw_os_error() == Some(ERROR_BROKEN_PIPE.0 as i32) {
                        self.done = true;
                        Ok(())
                    } else {
                        Err(e)
                    }
                }
            }
        }

        unsafe fn complete(&mut self, status: &CompletionStatus) {
            let prev = self.dst.len();
            self.dst.set_len(prev + status.bytes_transferred() as usize);
            if status.bytes_transferred() == 0 {
                self.done = true;
            }
        }
    }

    unsafe fn slice_to_end(v: &mut Vec<u8>) -> &mut [u8] {
        if v.capacity() == 0 {
            v.reserve(16);
        }
        if v.capacity() == v.len() {
            v.reserve(1);
        }
        slice::from_raw_parts_mut(v.as_mut_ptr().offset(v.len() as isize), v.capacity() - v.len())
    }
}
