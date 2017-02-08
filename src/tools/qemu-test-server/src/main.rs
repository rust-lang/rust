// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// This is a small server which is intended to run inside of an emulator. This
/// server pairs with the `qemu-test-client` program in this repository. The
/// `qemu-test-client` connects to this server over a TCP socket and performs
/// work such as:
///
/// 1. Pushing shared libraries to the server
/// 2. Running tests through the server
///
/// The server supports running tests concurrently and also supports tests
/// themselves having support libraries. All data over the TCP sockets is in a
/// basically custom format suiting our needs.

use std::fs::{self, File, Permissions};
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::net::{TcpListener, TcpStream};
use std::os::unix::prelude::*;
use std::sync::{Arc, Mutex};
use std::path::Path;
use std::str;
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT, Ordering};
use std::thread;
use std::process::{Command, Stdio};

macro_rules! t {
    ($e:expr) => (match $e {
        Ok(e) => e,
        Err(e) => panic!("{} failed with {}", stringify!($e), e),
    })
}

static TEST: AtomicUsize = ATOMIC_USIZE_INIT;

fn main() {
    println!("starting test server");
    let listener = t!(TcpListener::bind("10.0.2.15:12345"));
    println!("listening!");

    let work = Path::new("/tmp/work");
    t!(fs::create_dir_all(work));

    let lock = Arc::new(Mutex::new(()));

    for socket in listener.incoming() {
        let mut socket = t!(socket);
        let mut buf = [0; 4];
        t!(socket.read_exact(&mut buf));
        if &buf[..] == b"ping" {
            t!(socket.write_all(b"pong"));
        } else if &buf[..] == b"push" {
            handle_push(socket, work);
        } else if &buf[..] == b"run " {
            let lock = lock.clone();
            thread::spawn(move || handle_run(socket, work, &lock));
        } else {
            panic!("unknown command {:?}", buf);
        }
    }
}

fn handle_push(socket: TcpStream, work: &Path) {
    let mut reader = BufReader::new(socket);
    let mut filename = Vec::new();
    t!(reader.read_until(0, &mut filename));
    filename.pop(); // chop off the 0
    let filename = t!(str::from_utf8(&filename));

    let path = work.join(filename);
    t!(io::copy(&mut reader, &mut t!(File::create(&path))));
    t!(fs::set_permissions(&path, Permissions::from_mode(0o755)));
}

struct RemoveOnDrop<'a> {
    inner: &'a Path,
}

impl<'a> Drop for RemoveOnDrop<'a> {
    fn drop(&mut self) {
        t!(fs::remove_dir_all(self.inner));
    }
}

fn handle_run(socket: TcpStream, work: &Path, lock: &Mutex<()>) {
    let mut arg = Vec::new();
    let mut reader = BufReader::new(socket);

    // Allocate ourselves a directory that we'll delete when we're done to save
    // space.
    let n = TEST.fetch_add(1, Ordering::SeqCst);
    let path = work.join(format!("test{}", n));
    let exe = path.join("exe");
    t!(fs::create_dir(&path));
    let _a = RemoveOnDrop { inner: &path };

    // First up we'll get a list of arguments delimited with 0 bytes. An empty
    // argument means that we're done.
    let mut cmd = Command::new(&exe);
    while t!(reader.read_until(0, &mut arg)) > 1 {
        cmd.arg(t!(str::from_utf8(&arg[..arg.len() - 1])));
        arg.truncate(0);
    }

    // Next we'll get a bunch of env vars in pairs delimited by 0s as well
    arg.truncate(0);
    while t!(reader.read_until(0, &mut arg)) > 1 {
        let key_len = arg.len() - 1;
        let val_len = t!(reader.read_until(0, &mut arg)) - 1;
        {
            let key = &arg[..key_len];
            let val = &arg[key_len + 1..][..val_len];
            let key = t!(str::from_utf8(key));
            let val = t!(str::from_utf8(val));
            cmd.env(key, val);
        }
        arg.truncate(0);
    }

    // The section of code from here down to where we drop the lock is going to
    // be a critical section for us. On Linux you can't execute a file which is
    // open somewhere for writing, as you'll receive the error "text file busy".
    // Now here we never have the text file open for writing when we spawn it,
    // so why do we still need a critical section?
    //
    // Process spawning first involves a `fork` on Unix, which clones all file
    // descriptors into the child process. This means that it's possible for us
    // to open the file for writing (as we're downloading it), then some other
    // thread forks, then we close the file and try to exec. At that point the
    // other thread created a child process with the file open for writing, and
    // we attempt to execute it, so we get an error.
    //
    // This race is resolve by ensuring that only one thread can writ ethe file
    // and spawn a child process at once. Kinda an unfortunate solution, but we
    // don't have many other choices with this sort of setup!
    //
    // In any case the lock is acquired here, before we start writing any files.
    // It's then dropped just after we spawn the child. That way we don't lock
    // the execution of the child, just the creation of its files.
    let lock = lock.lock();

    // Next there's a list of dynamic libraries preceded by their filenames.
    arg.truncate(0);
    while t!(reader.read_until(0, &mut arg)) > 1 {
        let dst = path.join(t!(str::from_utf8(&arg[..arg.len() - 1])));
        let amt = read_u32(&mut reader) as u64;
        t!(io::copy(&mut reader.by_ref().take(amt),
                    &mut t!(File::create(&dst))));
        t!(fs::set_permissions(&dst, Permissions::from_mode(0o755)));
        arg.truncate(0);
    }

    // Finally we'll get the binary. The other end will tell us how big the
    // binary is and then we'll download it all to the exe path we calculated
    // earlier.
    let amt = read_u32(&mut reader) as u64;
    t!(io::copy(&mut reader.by_ref().take(amt),
                &mut t!(File::create(&exe))));
    t!(fs::set_permissions(&exe, Permissions::from_mode(0o755)));

    // Support libraries were uploaded to `work` earlier, so make sure that's
    // in `LD_LIBRARY_PATH`. Also include our own current dir which may have
    // had some libs uploaded.
    cmd.env("LD_LIBRARY_PATH",
            format!("{}:{}", work.display(), path.display()));

    // Spawn the child and ferry over stdout/stderr to the socket in a framed
    // fashion (poor man's style)
    let mut child = t!(cmd.stdin(Stdio::null())
                          .stdout(Stdio::piped())
                          .stderr(Stdio::piped())
                          .spawn());
    drop(lock);
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let socket = Arc::new(Mutex::new(reader.into_inner()));
    let socket2 = socket.clone();
    let thread = thread::spawn(move || my_copy(&mut stdout, 0, &*socket2));
    my_copy(&mut stderr, 1, &*socket);
    thread.join().unwrap();

    // Finally send over the exit status.
    let status = t!(child.wait());
    let (which, code) = match status.code() {
        Some(n) => (0, n),
        None => (1, status.signal().unwrap()),
    };
    t!(socket.lock().unwrap().write_all(&[
        which,
        (code >> 24) as u8,
        (code >> 16) as u8,
        (code >>  8) as u8,
        (code >>  0) as u8,
    ]));
}

fn my_copy(src: &mut Read, which: u8, dst: &Mutex<Write>) {
    let mut b = [0; 1024];
    loop {
        let n = t!(src.read(&mut b));
        let mut dst = dst.lock().unwrap();
        t!(dst.write_all(&[
            which,
            (n >> 24) as u8,
            (n >> 16) as u8,
            (n >>  8) as u8,
            (n >>  0) as u8,
        ]));
        if n > 0 {
            t!(dst.write_all(&b[..n]));
        } else {
            break
        }
    }
}

fn read_u32(r: &mut Read) -> u32 {
    let mut len = [0; 4];
    t!(r.read_exact(&mut len));
    ((len[0] as u32) << 24) |
    ((len[1] as u32) << 16) |
    ((len[2] as u32) <<  8) |
    ((len[3] as u32) <<  0)
}
