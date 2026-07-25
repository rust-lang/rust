//! Standalone stress reproducer for rust-lang/rust#124105.
//!
//! Hunts whatever closes a file descriptor it does not own on macOS, which
//! surfaces as either
//!
//!   panic: unexpected error during closedir "Bad file descriptor"
//!   fatal runtime error: IO Safety violation: owned file descriptor already closed
//!
//! There are two competing explanations and this program is built to tell them
//! apart rather than to assume one:
//!
//!   A. FSEvents (or CoreFoundation's CFURL resolution) closes a descriptor it
//!      does not own, and std is the victim.
//!   B. std's own `remove_dir_all` mishandles a descriptor on a large tree, and
//!      FSEvents is incidental.
//!
//! So there are three worker kinds, run concurrently by default:
//!
//!   faithful  mirrors notify's failing test as closely as possible: fresh 8194
//!             directory tree, per-path CFURL file-reference round trip, stream
//!             created with FileEvents|NoDefer|WatchRoot, a NEW thread per
//!             stream that creates a run loop, schedules, sees Start fail,
//!             invalidates, releases and exits, then remove_dir_all of THAT tree
//!   live      a stream that actually starts and is serviced by a running run
//!             loop, with its tree deleted underneath it
//!   control   remove_dir_all over the same size tree with NO FSEvents at all
//!
//! If `control` alone trips, explanation B is right and FSEvents is a red
//! herring. That is the single most useful result this program can produce.
//!
//! Detection does not rely on std's checks. It holds a pool of canaries and
//! checks each one for
//! * EBADF, confirmed with fcntl(F_GETFD) rather than fstat, because std's own
//!   comment notes fstat EBADF can be bubbled up from a FUSE server
//! * identity drift: a canary whose (dev, ino) changed had its descriptor closed
//!   by somebody else and the number handed to a new owner
//!
//! The pool is also rotated, so canaries keep occupying freshly recycled
//! descriptor numbers instead of sitting in a fixed low block forever.

use std::ffi::{c_char, c_int, c_void, CString};
use std::fs::File;
use std::os::unix::fs::MetadataExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

type CFIndex = isize;
type CFAllocatorRef = *const c_void;
type CFStringRef = *const c_void;
type CFArrayRef = *const c_void;
type CFMutableArrayRef = *mut c_void;
type CFURLRef = *const c_void;
type CFErrorRef = *mut c_void;
type CFRunLoopRef = *mut c_void;
type FSEventStreamRef = *mut c_void;
type FSEventStreamEventId = u64;
type FSEventStreamEventFlags = u32;
type FSEventStreamCreateFlags = u32;
type CFTimeInterval = f64;
type CFStringEncoding = u32;
type CFURLPathStyle = CFIndex;
type Boolean = u8;

const K_CF_STRING_ENCODING_UTF8: CFStringEncoding = 0x0800_0100;
const K_CF_URL_POSIX_PATH_STYLE: CFURLPathStyle = 0;
const K_FSEVENT_STREAM_EVENT_ID_SINCE_NOW: FSEventStreamEventId = 0xFFFF_FFFF_FFFF_FFFF;

// notify creates every stream with exactly these three, regardless of recursive
// mode. WatchRoot in particular makes FSEvents track each root by identity,
// which is the code most likely to hold per-path state.
const FLAG_NO_DEFER: FSEventStreamCreateFlags = 0x0000_0002;
const FLAG_WATCH_ROOT: FSEventStreamCreateFlags = 0x0000_0004;
const FLAG_FILE_EVENTS: FSEventStreamCreateFlags = 0x0000_0010;
const NOTIFY_FLAGS: FSEventStreamCreateFlags = FLAG_FILE_EVENTS | FLAG_NO_DEFER | FLAG_WATCH_ROOT;

const EBADF: i32 = 9;
const F_GETFD: c_int = 1;

#[repr(C)]
struct FSEventStreamContext {
    version: CFIndex,
    info: *mut c_void,
    retain: *const c_void,
    release: *const c_void,
    copy_description: *const c_void,
}

// The real 5-field layout rather than a zero-sized placeholder, so that taking a
// reference to the extern static is unambiguously the symbol address.
#[repr(C)]
struct CFArrayCallBacks {
    version: CFIndex,
    retain: *const c_void,
    release: *const c_void,
    copy_description: *const c_void,
    equal: *const c_void,
}

type FSEventStreamCallback = extern "C" fn(
    stream: FSEventStreamRef,
    info: *mut c_void,
    num_events: usize,
    event_paths: *mut c_void,
    event_flags: *const FSEventStreamEventFlags,
    event_ids: *const FSEventStreamEventId,
);

#[link(name = "CoreServices", kind = "framework")]
extern "C" {
    fn FSEventStreamCreate(
        allocator: CFAllocatorRef,
        callback: FSEventStreamCallback,
        context: *const FSEventStreamContext,
        paths_to_watch: CFArrayRef,
        since_when: FSEventStreamEventId,
        latency: CFTimeInterval,
        flags: FSEventStreamCreateFlags,
    ) -> FSEventStreamRef;
    fn FSEventStreamScheduleWithRunLoop(
        stream: FSEventStreamRef,
        runloop: CFRunLoopRef,
        mode: CFStringRef,
    );
    fn FSEventStreamStart(stream: FSEventStreamRef) -> Boolean;
    fn FSEventStreamStop(stream: FSEventStreamRef);
    fn FSEventStreamInvalidate(stream: FSEventStreamRef);
    fn FSEventStreamRelease(stream: FSEventStreamRef);
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    static kCFAllocatorDefault: CFAllocatorRef;
    static kCFRunLoopDefaultMode: CFStringRef;
    static kCFTypeArrayCallBacks: CFArrayCallBacks;

    fn CFStringCreateWithCString(
        alloc: CFAllocatorRef,
        c_str: *const c_char,
        encoding: CFStringEncoding,
    ) -> CFStringRef;
    fn CFArrayCreateMutable(
        alloc: CFAllocatorRef,
        capacity: CFIndex,
        callbacks: *const CFArrayCallBacks,
    ) -> CFMutableArrayRef;
    fn CFArrayAppendValue(array: CFMutableArrayRef, value: *const c_void);
    fn CFRelease(cf: *const c_void);
    fn CFRunLoopGetCurrent() -> CFRunLoopRef;
    fn CFRunLoopRunInMode(mode: CFStringRef, seconds: CFTimeInterval, return_after: Boolean) -> i32;

    fn CFURLCreateWithFileSystemPath(
        alloc: CFAllocatorRef,
        file_path: CFStringRef,
        path_style: CFURLPathStyle,
        is_directory: Boolean,
    ) -> CFURLRef;
    fn CFURLCopyAbsoluteURL(relative: CFURLRef) -> CFURLRef;
    fn CFURLResourceIsReachable(url: CFURLRef, error: *mut CFErrorRef) -> Boolean;
    fn CFURLCreateFileReferenceURL(
        alloc: CFAllocatorRef,
        url: CFURLRef,
        error: *mut CFErrorRef,
    ) -> CFURLRef;
    fn CFURLCreateFilePathURL(
        alloc: CFAllocatorRef,
        url: CFURLRef,
        error: *mut CFErrorRef,
    ) -> CFURLRef;
    fn CFURLCopyFileSystemPath(url: CFURLRef, path_style: CFURLPathStyle) -> CFStringRef;
}

extern "C" {
    // Declared variadic to match the real prototype; we pass no variadic args.
    fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
}

extern "C" fn noop_callback(
    _stream: FSEventStreamRef,
    _info: *mut c_void,
    _num_events: usize,
    _event_paths: *mut c_void,
    _event_flags: *const FSEventStreamEventFlags,
    _event_ids: *const FSEventStreamEventId,
) {
}

struct SendStream(FSEventStreamRef);
unsafe impl Send for SendStream {}

/// Mirror of notify's `path_to_cfstring_ref`: resolve the path through a file
/// reference URL and back. This is where CoreFoundation does its own opening and
/// resolving, once per path, and it is a prime suspect in its own right.
unsafe fn cfurl_round_trip(path: &Path) -> Option<CFStringRef> {
    let c = CString::new(path.to_str()?).ok()?;
    let s = CFStringCreateWithCString(kCFAllocatorDefault, c.as_ptr(), K_CF_STRING_ENCODING_UTF8);
    if s.is_null() {
        return None;
    }
    let url = CFURLCreateWithFileSystemPath(kCFAllocatorDefault, s, K_CF_URL_POSIX_PATH_STYLE, 1);
    CFRelease(s);
    if url.is_null() {
        return None;
    }
    let absolute = CFURLCopyAbsoluteURL(url);
    CFRelease(url);
    if absolute.is_null() {
        return None;
    }

    let mut err: CFErrorRef = std::ptr::null_mut();
    // The path exists, so unlike notify we do not need the imaginary-component
    // loop; if it is unreachable just bail.
    if CFURLResourceIsReachable(absolute, &mut err) == 0 {
        if !err.is_null() {
            CFRelease(err as *const c_void);
        }
        CFRelease(absolute);
        return None;
    }

    let reference = CFURLCreateFileReferenceURL(kCFAllocatorDefault, absolute, &mut err);
    CFRelease(absolute);
    if reference.is_null() {
        if !err.is_null() {
            CFRelease(err as *const c_void);
        }
        return None;
    }

    let back = CFURLCreateFilePathURL(kCFAllocatorDefault, reference, &mut err);
    CFRelease(reference);
    if back.is_null() {
        if !err.is_null() {
            CFRelease(err as *const c_void);
        }
        return None;
    }

    let out = CFURLCopyFileSystemPath(back, K_CF_URL_POSIX_PATH_STYLE);
    CFRelease(back);
    if out.is_null() {
        None
    } else {
        Some(out)
    }
}

/// Built fresh per cycle, like notify does, not once per worker.
unsafe fn build_cf_array(paths: &[PathBuf], round_trip: bool) -> CFMutableArrayRef {
    let array = CFArrayCreateMutable(kCFAllocatorDefault, 0, &kCFTypeArrayCallBacks);
    assert!(!array.is_null(), "CFArrayCreateMutable returned null");
    for path in paths {
        let s = if round_trip {
            match cfurl_round_trip(path) {
                Some(s) => s,
                None => continue,
            }
        } else {
            let c = CString::new(path.to_str().expect("utf-8")).expect("no nul");
            CFStringCreateWithCString(kCFAllocatorDefault, c.as_ptr(), K_CF_STRING_ENCODING_UTF8)
        };
        if s.is_null() {
            continue;
        }
        CFArrayAppendValue(array, s);
        CFRelease(s);
    }
    array
}

struct Counters {
    cycles: AtomicU64,
    start_failures: AtomicU64,
    create_nulls: AtomicU64,
    trees_removed: AtomicU64,
    remove_errors: AtomicU64,
    live_cycles: AtomicU64,
    control_removals: AtomicU64,
    rotations: AtomicU64,
}

impl Counters {
    fn new() -> Self {
        Self {
            cycles: AtomicU64::new(0),
            start_failures: AtomicU64::new(0),
            create_nulls: AtomicU64::new(0),
            trees_removed: AtomicU64::new(0),
            remove_errors: AtomicU64::new(0),
            live_cycles: AtomicU64::new(0),
            control_removals: AtomicU64::new(0),
            rotations: AtomicU64::new(0),
        }
    }
    fn line(&self) -> String {
        format!(
            "cycles={} start_fail={} create_null={} trees_rm={} rm_err={} live={} control={} rot={}",
            self.cycles.load(Ordering::Relaxed),
            self.start_failures.load(Ordering::Relaxed),
            self.create_nulls.load(Ordering::Relaxed),
            self.trees_removed.load(Ordering::Relaxed),
            self.remove_errors.load(Ordering::Relaxed),
            self.live_cycles.load(Ordering::Relaxed),
            self.control_removals.load(Ordering::Relaxed),
            self.rotations.load(Ordering::Relaxed),
        )
    }
}

fn build_tree(root: &Path, count: usize) -> std::io::Result<Vec<PathBuf>> {
    let mut leaves = Vec::with_capacity(count);
    for i in 0..count {
        let leaf = root.join(format!("dir_{i}/subdir"));
        std::fs::create_dir_all(&leaf)?;
        leaves.push(leaf);
    }
    Ok(leaves)
}

/// One faithful attempt. Returns true if Start failed, which is the intended path.
unsafe fn faithful_cycle(paths: &[PathBuf], counters: &Counters, round_trip: bool) -> bool {
    let array = build_cf_array(paths, round_trip);

    let context = FSEventStreamContext {
        version: 0,
        info: std::ptr::null_mut(),
        retain: std::ptr::null(),
        release: std::ptr::null(),
        copy_description: std::ptr::null(),
    };
    let stream = FSEventStreamCreate(
        kCFAllocatorDefault,
        noop_callback,
        &context,
        array as CFArrayRef,
        K_FSEVENT_STREAM_EVENT_ID_SINCE_NOW,
        0.0,
        NOTIFY_FLAGS,
    );
    if stream.is_null() {
        counters.create_nulls.fetch_add(1, Ordering::Relaxed);
        CFRelease(array as *const c_void);
        return false;
    }

    // A fresh thread per stream, which creates a run loop, schedules, watches
    // Start fail, invalidates, releases, and exits, destroying the run loop.
    // notify does exactly this and the teardown is itself a suspect.
    let send = SendStream(stream);
    let failed = std::thread::spawn(move || {
        let send = send;
        FSEventStreamScheduleWithRunLoop(send.0, CFRunLoopGetCurrent(), kCFRunLoopDefaultMode);
        let started = FSEventStreamStart(send.0);
        if started != 0 {
            FSEventStreamStop(send.0);
        }
        FSEventStreamInvalidate(send.0);
        FSEventStreamRelease(send.0);
        started == 0
    })
    .join()
    .expect("fsevents thread panicked");

    CFRelease(array as *const c_void);
    counters.cycles.fetch_add(1, Ordering::Relaxed);
    if failed {
        counters.start_failures.fetch_add(1, Ordering::Relaxed);
    }
    failed
}

struct Canary {
    file: File,
    dev: u64,
    ino: u64,
    path: PathBuf,
}

impl Canary {
    fn open(path: PathBuf) -> std::io::Result<Self> {
        let file = File::open(&path)?;
        let m = file.metadata()?;
        Ok(Self { dev: m.dev(), ino: m.ino(), file, path })
    }
}

fn report(what: &str, index: usize, counters: &Counters) -> ! {
    use std::io::Write;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out);
    let _ = writeln!(out, "REPRODUCED: {what}");
    let _ = writeln!(out, "canary index {index}");
    let _ = writeln!(out, "{}", counters.line());
    let _ = writeln!(
        out,
        "nothing in this program closes canary descriptors or reopens them out of band"
    );
    let _ = out.flush();
    // Skip atexit handlers: other threads are inside CoreFoundation right now
    // and running __cxa_finalize under them tends to crash confusingly.
    std::process::exit(1)
}

fn main() {
    let paths: usize = env("PATHS", 4097);
    let canary_count: usize = env("CANARIES", 64);
    let seconds: u64 = env("SECONDS", 300);
    let faithful_workers: usize = env("FAITHFUL", 1);
    let live_workers: usize = env("LIVE", 1);
    let control_workers: usize = env("CONTROL", 1);
    let round_trip: bool = env::<u8>("CFURL", 1) != 0;

    let root = std::env::temp_dir().join(format!("fsevents-fd-repro-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create root");
    // notify hands FSEvents canonical paths, so match that.
    let root = root.canonicalize().expect("canonicalize root");

    println!("paths={paths} canaries={canary_count} seconds={seconds}");
    println!(
        "workers: faithful={faithful_workers} live={live_workers} control={control_workers} cfurl_round_trip={round_trip}"
    );
    println!("if only the control worker trips, the bug is in std and FSEvents is incidental");

    let panicked = Arc::new(AtomicBool::new(false));
    {
        let panicked = Arc::clone(&panicked);
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            panicked.store(true, Ordering::SeqCst);
            previous(info);
        }));
    }

    let canary_dir = root.join("canaries");
    std::fs::create_dir_all(&canary_dir).expect("create canary dir");
    for i in 0..canary_count {
        std::fs::write(canary_dir.join(format!("canary_{i}")), b"canary").expect("write canary");
    }

    let counters = Arc::new(Counters::new());
    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();

    // Canary monitor. Rotates part of the pool each sweep so canaries keep
    // landing on freshly recycled descriptor numbers rather than sitting in a
    // fixed low block that nothing ever reuses.
    {
        let stop = Arc::clone(&stop);
        let counters = Arc::clone(&counters);
        let canary_dir = canary_dir.clone();
        handles.push(std::thread::spawn(move || {
            let mut canaries: Vec<Canary> = (0..canary_count)
                .map(|i| Canary::open(canary_dir.join(format!("canary_{i}"))).expect("open canary"))
                .collect();
            let mut sweep: usize = 0;
            while !stop.load(Ordering::Relaxed) {
                for (i, c) in canaries.iter().enumerate() {
                    let fd = c.file.as_raw_fd();
                    // Authoritative EBADF probe: queries the process descriptor
                    // table, so unlike fstat it cannot be bubbled up by a
                    // userspace filesystem server.
                    if unsafe { fcntl(fd, F_GETFD) } == -1 {
                        let e = std::io::Error::last_os_error();
                        if e.raw_os_error() == Some(EBADF) {
                            report("canary descriptor was closed by something else", i, &counters);
                        }
                    }
                    match c.file.metadata() {
                        Ok(m) => {
                            if m.dev() != c.dev || m.ino() != c.ino {
                                report(
                                    "canary descriptor now refers to a different file, so it was closed and the number reused",
                                    i,
                                    &counters,
                                );
                            }
                        }
                        Err(e) => {
                            if e.raw_os_error() != Some(EBADF) {
                                eprintln!("canary {i}: unexpected error {e:?}");
                            }
                        }
                    }
                }
                // Rotate a slice of the pool.
                let n = canaries.len();
                if n > 0 {
                    let i = sweep % n;
                    let path = canaries[i].path.clone();
                    canaries[i] = Canary::open(path).expect("reopen canary");
                    counters.rotations.fetch_add(1, Ordering::Relaxed);
                }
                sweep = sweep.wrapping_add(1);
            }
        }));
    }

    // Faithful workers.
    for w in 0..faithful_workers {
        let root = root.clone();
        let stop = Arc::clone(&stop);
        let counters = Arc::clone(&counters);
        handles.push(std::thread::spawn(move || {
            let mut n = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let tree = root.join(format!("faithful_{w}_{n}"));
                if std::fs::create_dir_all(&tree).is_err() {
                    break;
                }
                let leaves = match build_tree(&tree, paths) {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("faithful worker: build_tree failed: {e:?}");
                        break;
                    }
                };
                unsafe { faithful_cycle(&leaves, &counters, round_trip) };
                // Delete the very tree FSEvents was just given. This coupling is
                // the whole hypothesis; without it the workers share nothing.
                match std::fs::remove_dir_all(&tree) {
                    Ok(()) => counters.trees_removed.fetch_add(1, Ordering::Relaxed),
                    Err(_) => counters.remove_errors.fetch_add(1, Ordering::Relaxed),
                };
                n += 1;
            }
        }));
    }

    // Live workers: a stream that actually starts, is serviced by a running run
    // loop, and has its tree deleted underneath it.
    for w in 0..live_workers {
        let root = root.clone();
        let stop = Arc::clone(&stop);
        let counters = Arc::clone(&counters);
        handles.push(std::thread::spawn(move || {
            let mut n = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let tree = root.join(format!("live_{w}_{n}"));
                if std::fs::create_dir_all(&tree).is_err() {
                    break;
                }
                let leaves = match build_tree(&tree, 500) {
                    Ok(l) => l,
                    Err(_) => break,
                };
                unsafe {
                    let array = build_cf_array(&leaves, round_trip);
                    let context = FSEventStreamContext {
                        version: 0,
                        info: std::ptr::null_mut(),
                        retain: std::ptr::null(),
                        release: std::ptr::null(),
                        copy_description: std::ptr::null(),
                    };
                    let stream = FSEventStreamCreate(
                        kCFAllocatorDefault,
                        noop_callback,
                        &context,
                        array as CFArrayRef,
                        K_FSEVENT_STREAM_EVENT_ID_SINCE_NOW,
                        0.0,
                        NOTIFY_FLAGS,
                    );
                    CFRelease(array as *const c_void);
                    if stream.is_null() {
                        counters.create_nulls.fetch_add(1, Ordering::Relaxed);
                    } else {
                        FSEventStreamScheduleWithRunLoop(
                            stream,
                            CFRunLoopGetCurrent(),
                            kCFRunLoopDefaultMode,
                        );
                        if FSEventStreamStart(stream) != 0 {
                            // Service the run loop briefly so the stream is
                            // genuinely live, then delete underneath it.
                            CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.05, 0);
                            let _ = std::fs::remove_dir_all(&tree);
                            CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.05, 0);
                            FSEventStreamStop(stream);
                            counters.live_cycles.fetch_add(1, Ordering::Relaxed);
                        }
                        FSEventStreamInvalidate(stream);
                        FSEventStreamRelease(stream);
                    }
                }
                let _ = std::fs::remove_dir_all(&tree);
                n += 1;
            }
        }));
    }

    // Control workers: identical tree churn with no FSEvents anywhere.
    for w in 0..control_workers {
        let root = root.clone();
        let stop = Arc::clone(&stop);
        let counters = Arc::clone(&counters);
        handles.push(std::thread::spawn(move || {
            let mut n = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let tree = root.join(format!("control_{w}_{n}"));
                if std::fs::create_dir_all(&tree).is_err() {
                    break;
                }
                if build_tree(&tree, paths).is_err() {
                    break;
                }
                match std::fs::remove_dir_all(&tree) {
                    Ok(()) => counters.control_removals.fetch_add(1, Ordering::Relaxed),
                    Err(_) => counters.remove_errors.fetch_add(1, Ordering::Relaxed),
                };
                n += 1;
            }
        }));
    }

    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut checked_progress = false;
    while Instant::now() < deadline && !panicked.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_secs(2));
        use std::io::Write;
        print!("\r{}   ", counters.line());
        let _ = std::io::stdout().flush();

        // A run where Start never fails, or Create always returns null, is
        // exercising nothing. Say so early rather than after the full run.
        if !checked_progress && counters.cycles.load(Ordering::Relaxed) >= 3 {
            checked_progress = true;
            if counters.start_failures.load(Ordering::Relaxed) == 0 {
                println!();
                println!("WARNING: FSEventStreamStart never failed. Either PATHS is below the cap");
                println!("or the FFI is wrong. The faithful worker is not exercising its path.");
            }
            if counters.create_nulls.load(Ordering::Relaxed) > 0 {
                println!();
                println!("WARNING: FSEventStreamCreate returned null, so those cycles did nothing.");
            }
        }
    }

    stop.store(true, Ordering::Relaxed);
    for h in handles {
        if h.join().is_err() {
            panicked.store(true, Ordering::SeqCst);
        }
    }
    let _ = std::fs::remove_dir_all(&root);

    println!();
    println!("{}", counters.line());
    if panicked.load(Ordering::SeqCst) {
        println!("a worker PANICKED, which is very likely the bug. Scroll up for the message.");
        std::process::exit(1);
    }
    println!("no stray close observed. This is a race, so a clean run is weak evidence.");
}

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(v) => match v.parse() {
            Ok(parsed) => parsed,
            Err(_) => {
                eprintln!("warning: {name}={v:?} could not be parsed, using the default");
                default
            }
        },
        Err(_) => default,
    }
}
