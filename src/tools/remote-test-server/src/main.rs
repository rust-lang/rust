//! This is a small server which is intended to run inside of an emulator or
//! on a remote test device. This server pairs with the `remote-test-client`
//! program in this repository. The `remote-test-client` connects to this
//! server over a TCP socket and performs work such as:
//!
//! 1. Pushing shared libraries to the server
//! 2. Running tests through the server
//!
//! The server supports running tests concurrently and also supports tests
//! themselves having support libraries. All data over the TCP sockets is in a
//! basically custom format suiting our needs.

#[cfg(not(windows))]
use std::fs::Permissions;
use std::fs::{self, File};
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::net::{SocketAddr, TcpListener, TcpStream};
#[cfg(not(windows))]
use std::os::unix::prelude::*;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::{cmp, env, str, thread};

macro_rules! t {
    ($e:expr) => {
        match $e {
            Ok(e) => e,
            Err(e) => panic!("{} failed with {}", stringify!($e), e),
        }
    };
}

static TEST: AtomicUsize = AtomicUsize::new(0);
const RETRY_INTERVAL: u64 = 1;
const NUMBER_OF_RETRIES: usize = 5;

#[derive(Copy, Clone)]
struct Config {
    verbose: bool,
    sequential: bool,
    batch: bool,
    bind: SocketAddr,
}

impl Config {
    pub fn default() -> Config {
        Config {
            verbose: false,
            sequential: false,
            batch: false,
            bind: if cfg!(target_os = "android") || cfg!(windows) {
                ([0, 0, 0, 0], 12345).into()
            } else {
                ([10, 0, 2, 15], 12345).into()
            },
        }
    }

    pub fn parse_args() -> Config {
        let mut config = Config::default();

        let args = env::args().skip(1);
        let mut next_is_bind = false;
        for argument in args {
            match &argument[..] {
                bind if next_is_bind => {
                    config.bind = t!(bind.parse());
                    next_is_bind = false;
                }
                "--bind" => next_is_bind = true,
                "--sequential" => config.sequential = true,
                "--batch" => config.batch = true,
                "--verbose" | "-v" => config.verbose = true,
                "--help" | "-h" => {
                    show_help();
                    std::process::exit(0);
                }
                arg => panic!("unknown argument: {}, use `--help` for known arguments", arg),
            }
        }
        if next_is_bind {
            panic!("missing value for --bind");
        }

        config
    }
}

fn show_help() {
    eprintln!(
        r#"Usage:

{} [OPTIONS]

OPTIONS:
    --bind <IP>:<PORT>   Specify IP address and port to listen for requests, e.g. "0.0.0.0:12345"
    --sequential         Run only one test at a time
    --batch              Send stdout and stderr in batch instead of streaming
    -v, --verbose        Show status messages
    -h, --help           Show this help screen
"#,
        std::env::args().next().unwrap()
    );
}

fn print_verbose(s: &str, conf: Config) {
    if conf.verbose {
        println!("{}", s);
    }
}

fn main() {
    let config = Config::parse_args();
    println!("starting test server");

    let listener = bind_socket(config.bind);
    let (work, tmp): (PathBuf, PathBuf) = if cfg!(target_os = "android") {
        ("/data/local/tmp/work".into(), "/data/local/tmp/work/tmp".into())
    } else {
        let mut work_dir = env::temp_dir();
        work_dir.push("work");
        let mut tmp_dir = work_dir.clone();
        tmp_dir.push("tmp");
        (work_dir, tmp_dir)
    };
    println!("listening on {}!", config.bind);

    t!(fs::create_dir_all(&work));
    t!(fs::create_dir_all(&tmp));

    let lock = Arc::new(Mutex::new(()));

    for socket in listener.incoming() {
        let mut socket = t!(socket);
        let mut buf = [0; 4];
        if socket.read_exact(&mut buf).is_err() {
            continue;
        }
        if &buf[..] == b"ping" {
            print_verbose("Received ping", config);
            t!(socket.write_all(b"pong"));
        } else if &buf[..] == b"push" {
            handle_push(socket, &work, config);
        } else if &buf[..] == b"run " {
            let lock = lock.clone();
            let work = work.clone();
            let tmp = tmp.clone();
            let f = move || handle_run(socket, &work, &tmp, &lock, config);
            if config.sequential {
                f();
            } else {
                thread::spawn(f);
            }
        } else {
            panic!("unknown command {:?}", buf);
        }
    }
}

fn bind_socket(addr: SocketAddr) -> TcpListener {
    for _ in 0..(NUMBER_OF_RETRIES - 1) {
        if let Ok(x) = TcpListener::bind(addr) {
            return x;
        }
        std::thread::sleep(std::time::Duration::from_secs(RETRY_INTERVAL));
    }
    TcpListener::bind(addr).unwrap()
}

fn handle_push(socket: TcpStream, work: &Path, config: Config) {
    let mut reader = BufReader::new(socket);
    let dst = recv(&work, &mut reader);
    print_verbose(&format!("push {:#?}", dst), config);

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

fn handle_run(socket: TcpStream, work: &Path, tmp: &Path, lock: &Mutex<()>, config: Config) {
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
    print_verbose(&format!("run {:#?}", exe), config);

    let mut cmd = Command::new(&exe);
    cmd.args(args);
    cmd.envs(env);

    // On windows, libraries are just searched in the executable directory,
    // system directories, PWD, and PATH, in that order. PATH is the only one
    // we can change for this.
    let library_path = if cfg!(windows) { "PATH" } else { "LD_LIBRARY_PATH" };

    // Support libraries were uploaded to `work` earlier, so make sure that's
    // in `LD_LIBRARY_PATH`. Also include our own current dir which may have
    // had some libs uploaded.
    let mut paths = vec![work.to_owned(), path.clone()];
    if let Some(library_path) = env::var_os(library_path) {
        paths.extend(env::split_paths(&library_path));
    }
    cmd.env(library_path, env::join_paths(paths).unwrap());

    // Some tests assume RUST_TEST_TMPDIR exists
    cmd.env("RUST_TEST_TMPDIR", tmp);

    let socket = Arc::new(Mutex::new(reader.into_inner()));

    let status = if config.batch {
        let child =
            t!(cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).output());
        batch_copy(&child.stdout, 0, &*socket);
        batch_copy(&child.stderr, 1, &*socket);
        child.status
    } else {
        // Spawn the child and ferry over stdout/stderr to the socket in a framed
        // fashion (poor man's style)
        let mut child =
            t!(cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn());
        drop(lock);
        let mut stdout = child.stdout.take().unwrap();
        let mut stderr = child.stderr.take().unwrap();
        let socket2 = socket.clone();
        let thread = thread::spawn(move || my_copy(&mut stdout, 0, &*socket2));
        my_copy(&mut stderr, 1, &*socket);
        thread.join().unwrap();
        t!(child.wait())
    };

    // Finally send over the exit status.
    let (which, code) = get_status_code(&status);

    t!(socket.lock().unwrap().write_all(&[
        which,
        (code >> 24) as u8,
        (code >> 16) as u8,
        (code >> 8) as u8,
        (code >> 0) as u8,
    ]));
}

#[cfg(not(windows))]
fn get_status_code(status: &ExitStatus) -> (u8, i32) {
    match status.code() {
        Some(n) => (0, n),
        None => (1, status.signal().unwrap()),
    }
}

#[cfg(windows)]
fn get_status_code(status: &ExitStatus) -> (u8, i32) {
    (0, status.code().unwrap())
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
    let amt = read_u64(io);
    t!(io::copy(&mut io.take(amt), &mut t!(File::create(&dst))));
    set_permissions(&dst);
    dst
}

#[cfg(not(windows))]
fn set_permissions(path: &Path) {
    t!(fs::set_permissions(&path, Permissions::from_mode(0o755)));
}
#[cfg(windows)]
fn set_permissions(_path: &Path) {}

fn my_copy(src: &mut dyn Read, which: u8, dst: &Mutex<dyn Write>) {
    let mut b = [0; 1024];
    loop {
        let n = t!(src.read(&mut b));
        let mut dst = dst.lock().unwrap();
        t!(dst.write_all(&create_header(which, n as u64)));
        if n > 0 {
            t!(dst.write_all(&b[..n]));
        } else {
            break;
        }
    }
}

fn batch_copy(buf: &[u8], which: u8, dst: &Mutex<dyn Write>) {
    let n = buf.len();
    let mut dst = dst.lock().unwrap();
    t!(dst.write_all(&create_header(which, n as u64)));
    if n > 0 {
        t!(dst.write_all(buf));
        // Marking buf finished
        t!(dst.write_all(&[which, 0, 0, 0, 0,]));
    }
}

const fn create_header(which: u8, n: u64) -> [u8; 9] {
    let bytes = n.to_be_bytes();
    [which, bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]]
}

fn read_u64(r: &mut dyn Read) -> u64 {
    let mut len = [0; 8];
    t!(r.read_exact(&mut len));
    u64::from_be_bytes(len)
}
