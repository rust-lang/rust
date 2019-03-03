#![deny(rust_2018_idioms)]

/// This is a small server which is intended to run inside of an emulator or
/// on a remote test device. This server pairs with the `remote-test-client`
/// program in this repository. The `remote-test-client` connects to this
/// server over a TCP socket and performs work such as:
///
/// 1. Pushing shared libraries to the server
/// 2. Running tests through the server
///
/// The server supports running tests concurrently and also supports tests
/// themselves having support libraries. All data over the TCP sockets is in a
/// basically custom format suiting our needs.

use std::cmp;
use std::env;
use std::fs::{self, File, Permissions};
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::net::{TcpListener, TcpStream};
use std::os::unix::prelude::*;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::str;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

macro_rules! t {
    ($e:expr) => (match $e {
        Ok(e) => e,
        Err(e) => panic!("{} failed with {}", stringify!($e), e),
    })
}

static TEST: AtomicUsize = AtomicUsize::new(0);

struct Config {
    pub remote: bool,
    pub verbose: bool,
}

impl Config {
    pub fn default() -> Config {
        Config {
            remote: false,
            verbose: false,
        }
    }

    pub fn parse_args() -> Config {
        let mut config = Config::default();

        let args = env::args().skip(1);
        for argument in args {
            match &argument[..] {
                "remote" => {
                    config.remote = true;
                },
                "verbose" | "-v" => {
                    config.verbose = true;
                }
                arg => panic!("unknown argument: {}", arg),
            }
        }

        config
    }
}

fn main() {
    println!("starting test server");

    let config = Config::parse_args();

    let bind_addr = if cfg!(target_os = "android") || config.remote {
        "0.0.0.0:12345"
    } else {
        "10.0.2.15:12345"
    };

    let (listener, work) = if cfg!(target_os = "android") {
        (t!(TcpListener::bind(bind_addr)), "/data/tmp/work")
    } else {
        (t!(TcpListener::bind(bind_addr)), "/tmp/work")
    };
    println!("listening!");

    let work = Path::new(work);
    t!(fs::create_dir_all(work));

    let lock = Arc::new(Mutex::new(()));

    for socket in listener.incoming() {
        let mut socket = t!(socket);
        let mut buf = [0; 4];
        if socket.read_exact(&mut buf).is_err() {
            continue
        }
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
    recv(&work, &mut reader);

    let mut socket = reader.into_inner();
    t!(socket.write_all(b"ack "));
}

struct RemoveOnDrop<'a> {
    inner: &'a Path,
}

impl Drop for RemoveOnDrop<'_> {
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
    t!(fs::create_dir(&path));
    let _a = RemoveOnDrop { inner: &path };

    // First up we'll get a list of arguments delimited with 0 bytes. An empty
    // argument means that we're done.
    let mut args = Vec::new();
    while t!(reader.read_until(0, &mut arg)) > 1 {
        args.push(t!(str::from_utf8(&arg[..arg.len() - 1])).to_string());
        arg.truncate(0);
    }

    // Next we'll get a bunch of env vars in pairs delimited by 0s as well
    let mut env = Vec::new();
    arg.truncate(0);
    while t!(reader.read_until(0, &mut arg)) > 1 {
        let key_len = arg.len() - 1;
        let val_len = t!(reader.read_until(0, &mut arg)) - 1;
        {
            let key = &arg[..key_len];
            let val = &arg[key_len + 1..][..val_len];
            let key = t!(str::from_utf8(key)).to_string();
            let val = t!(str::from_utf8(val)).to_string();
            env.push((key, val));
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
    // This race is resolve by ensuring that only one thread can write the file
    // and spawn a child process at once. Kinda an unfortunate solution, but we
    // don't have many other choices with this sort of setup!
    //
    // In any case the lock is acquired here, before we start writing any files.
    // It's then dropped just after we spawn the child. That way we don't lock
    // the execution of the child, just the creation of its files.
    let lock = lock.lock();

    // Next there's a list of dynamic libraries preceded by their filenames.
    while t!(reader.fill_buf())[0] != 0 {
        recv(&path, &mut reader);
    }
    assert_eq!(t!(reader.read(&mut [0])), 1);

    // Finally we'll get the binary. The other end will tell us how big the
    // binary is and then we'll download it all to the exe path we calculated
    // earlier.
    let exe = recv(&path, &mut reader);

    let mut cmd = Command::new(&exe);
    for arg in args {
        cmd.arg(arg);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }

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

fn recv<B: BufRead>(dir: &Path, io: &mut B) -> PathBuf {
    let mut filename = Vec::new();
    t!(io.read_until(0, &mut filename));

    // We've got some tests with *really* long names. We try to name the test
    // executable the same on the target as it is on the host to aid with
    // debugging, but the targets we're emulating are often more restrictive
    // than the hosts as well.
    //
    // To ensure we can run a maximum number of tests without modifications we
    // just arbitrarily truncate the filename to 50 bytes. That should
    // hopefully allow us to still identify what's running while staying under
    // the filesystem limits.
    let len = cmp::min(filename.len() - 1, 50);
    let dst = dir.join(t!(str::from_utf8(&filename[..len])));
    let amt = read_u32(io) as u64;
    t!(io::copy(&mut io.take(amt),
                &mut t!(File::create(&dst))));
    t!(fs::set_permissions(&dst, Permissions::from_mode(0o755)));
    dst
}

fn my_copy(src: &mut dyn Read, which: u8, dst: &Mutex<dyn Write>) {
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

fn read_u32(r: &mut dyn Read) -> u32 {
    let mut len = [0; 4];
    t!(r.read_exact(&mut len));
    ((len[0] as u32) << 24) |
    ((len[1] as u32) << 16) |
    ((len[2] as u32) <<  8) |
    ((len[3] as u32) <<  0)
}
