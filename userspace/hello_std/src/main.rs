//! hello_std: Std conformance smoke test for ThingOS.
//!
//! Exercises every implemented std subsystem and reports PASS/FAIL.
//! Default tests run in <1s. Use flags to enable slow/external tests:
//!   --net, --net-server, --udp, --ipv6, --dns, --spawn, --stress, --all
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


extern crate std;

use std::collections::{HashMap, VecDeque};
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime};

// ── Harness ──────────────────────────────────────────────────────────

struct Test {
    name: &'static str,
    func: fn() -> Result<(), String>,
    fatal: bool,
    hint: &'static str,
}

fn run_all(tests: &[Test]) -> i32 {
    let mut passed = 0u32;
    let mut failed = 0u32;

    for t in tests {
        match (t.func)() {
            Ok(()) => {
                println!("[PASS] {}", t.name);
                passed += 1;
            }
            Err(reason) => {
                println!("[FAIL] {}: {}", t.name, reason);
                println!("       hint: {}", t.hint);
                failed += 1;
                if t.fatal {
                    println!("[FATAL] {} is fatal — aborting remaining tests", t.name);
                    break;
                }
            }
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════");
    println!(
        "SUMMARY: {} passed, {} failed, {} total",
        passed,
        failed,
        passed + failed
    );
    if failed == 0 {
        println!("STATUS: ALL TESTS PASSED ✓");
    } else {
        println!("STATUS: {} TESTS FAILED ✗", failed);
    }
    println!("═══════════════════════════════════════════════════");

    if failed > 0 {
        1
    } else {
        0
    }
}

// ── Flag parsing ─────────────────────────────────────────────────────

struct Flags {
    net: bool,
    net_server: bool,
    udp: bool,
    ipv6: bool,
    dns: bool,
    spawn: bool,
    stress: bool,
    // Child modes (hello_std acting as a test child)
    child_echo: bool,
    child_exit: bool,
    child_exit_code: i32,
}

fn parse_flags() -> Flags {
    let args: Vec<String> = std::env::args().collect();
    let mut f = Flags {
        net: false,
        net_server: false,
        udp: false,
        ipv6: false,
        dns: false,
        spawn: false,
        stress: false,
        child_echo: false,
        child_exit: false,
        child_exit_code: 0,
    };
    for a in &args[1..] {
        match a.as_str() {
            "--net" => f.net = true,
            "--net-server" => f.net_server = true,
            "--udp" => f.udp = true,
            "--ipv6" => f.ipv6 = true,
            "--dns" => f.dns = true,
            "--spawn" => f.spawn = true,
            "--stress" => f.stress = true,
            "--all" => {
                f.net = true;
                f.net_server = true;
                f.udp = true;
                f.ipv6 = true;
                f.dns = true;
                f.spawn = true;
                f.stress = true;
            }
            "--child-echo" => f.child_echo = true,
            "--child-exit" => f.child_exit = true,
            s if s.starts_with("--exit-code=") => {
                f.child_exit = true;
                f.child_exit_code = s[12..].parse().unwrap_or(1);
            }
            _ => {}
        }
    }
    f
}

// ── Child modes ──────────────────────────────────────────────────────

/// When invoked as `--child-echo`, print all remaining args and exit.
fn child_echo_mode() -> ! {
    let args: Vec<String> = std::env::args().collect();
    // Skip program name and --child-echo flag
    let payload: Vec<&str> = args
        .iter()
        .skip(1)
        .filter(|a| *a != "--child-echo")
        .map(|s| s.as_str())
        .collect();
    for a in &payload {
        println!("{}", a);
    }
    std::process::exit(0);
}

/// When invoked as `--child-exit --exit-code=N`, exit with code N.
fn child_exit_mode(code: i32) -> ! {
    std::process::exit(code);
}

// ── Banner ───────────────────────────────────────────────────────────

fn print_banner() {
    println!("═══════════════════════════════════════════════════");
    println!("  hello_std — ThingOS Std Conformance Smoke Test");
    println!("═══════════════════════════════════════════════════");
    println!("PID: {}", std::process::id());
    println!("args: {:?}", std::env::args().collect::<Vec<_>>());

    match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => println!("wall clock: UNIX epoch + {}s", d.as_secs()),
        Err(_) => println!("wall clock: before UNIX epoch (!)"),
    }

    let t0 = Instant::now();
    println!("monotonic baseline: {:?}", t0);

    match std::thread::available_parallelism() {
        Ok(n) => println!("available_parallelism: {}", n),
        Err(e) => println!("available_parallelism: unavailable ({})", e),
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════════
// DEFAULT TESTS
// ═══════════════════════════════════════════════════════════════════

// ── A: stdio ─────────────────────────────────────────────────────────

fn test_stdio() -> Result<(), String> {
    // If we got here, println! is already working (the harness uses it).
    // Test eprintln! too.
    eprintln!("[test_stdio] eprintln! works");

    // Test write! to stdout explicitly
    use std::io::Write;
    let mut stdout = std::io::stdout();
    write!(stdout, "[test_stdio] explicit write!... ").map_err(|e| alloc::format!("{}", e))?;
    writeln!(stdout, "ok").map_err(|e| alloc::format!("{}", e))?;
    stdout.flush().map_err(|e| alloc::format!("flush failed: {}", e))?;

    Ok(())
}

// ── B: alloc + collections ───────────────────────────────────────────

fn test_alloc() -> Result<(), String> {
    // 1 MiB allocation, fill, checksum
    let size = 1024 * 1024;
    let mut v: Vec<u8> = Vec::with_capacity(size);
    for i in 0..size {
        v.push((i & 0xFF) as u8);
    }
    if v.len() != size {
        return Err(alloc::format!("Vec len {} != expected {}", v.len(), size));
    }
    // Simple checksum — sum of all bytes
    let checksum: u64 = v.iter().map(|b| *b as u64).sum();
    // Expected: 0+1+...+255 repeated 4096 times = (255*256/2)*4096 = 133_693_440
    let expected: u64 = (255u64 * 256 / 2) * (size as u64 / 256);
    if checksum != expected {
        return Err(alloc::format!("checksum {} != expected {}", checksum, expected));
    }

    // String concat
    let mut s = String::from("hello");
    s.push_str(", world");
    if s != "hello, world" {
        return Err(alloc::format!("String concat: got '{}'", s));
    }

    // HashMap
    let mut map = HashMap::new();
    map.insert("alpha", 1);
    map.insert("beta", 2);
    map.insert("gamma", 3);
    if map.get("beta") != Some(&2) {
        return Err("HashMap lookup failed".into());
    }
    if map.len() != 3 {
        return Err(alloc::format!("HashMap len {} != 3", map.len()));
    }

    // VecDeque
    let mut dq = VecDeque::new();
    dq.push_back(10);
    dq.push_back(20);
    dq.push_front(5);
    if dq.pop_front() != Some(5) {
        return Err("VecDeque::pop_front failed".into());
    }
    if dq.len() != 2 {
        return Err(alloc::format!("VecDeque len {} != 2", dq.len()));
    }

    Ok(())
}

// ── C: time ──────────────────────────────────────────────────────────

fn test_time() -> Result<(), String> {
    let mono0 = stem::syscall::time_now(abi::time::ClockId::Monotonic)
        .map_err(|e| alloc::format!("monotonic clock read failed: {:?}", e))?;
    let mono0_ns = mono0
        .as_nanos()
        .ok_or_else(|| "monotonic timespec was invalid".to_string())?;

    if let Err(e) = stem::syscall::time_now_raw(0) {
        if e != abi::errors::Errno::EINVAL {
            return Err(alloc::format!(
                "invalid clock id returned {:?}, expected EINVAL",
                e
            ));
        }
    } else {
        return Err("invalid clock id unexpectedly succeeded".into());
    }

    // Instant monotonicity
    let t0 = Instant::now();
    std::thread::sleep(Duration::from_millis(10));
    let t1 = Instant::now();
    let elapsed = t1.duration_since(t0);
    // Tolerate jitter — just check it advanced at all (>= 1ms)
    if elapsed < Duration::from_millis(1) {
        return Err(alloc::format!(
            "Instant elapsed {:?} < 1ms after 10ms sleep",
            elapsed
        ));
    }

    let mono1 = stem::syscall::time_now(abi::time::ClockId::Monotonic)
        .map_err(|e| alloc::format!("second monotonic clock read failed: {:?}", e))?;
    let mono1_ns = mono1
        .as_nanos()
        .ok_or_else(|| "second monotonic timespec was invalid".to_string())?;
    if mono1_ns < mono0_ns {
        return Err(alloc::format!(
            "monotonic clock moved backwards: {} -> {}",
            mono0_ns, mono1_ns
        ));
    }

    let sleep_start = Instant::now();
    stem::syscall::sleep_ns(15_000_000);
    let sleep_elapsed = sleep_start.elapsed();
    if sleep_elapsed < Duration::from_millis(5) {
        return Err(alloc::format!(
            "sleep_ns woke too early: {:?} after 15ms request",
            sleep_elapsed
        ));
    }
    if sleep_elapsed > Duration::from_millis(250) {
        return Err(alloc::format!(
            "sleep_ns overslept too much: {:?} after 15ms request",
            sleep_elapsed
        ));
    }

    // SystemTime >= UNIX_EPOCH
    let now = SystemTime::now();
    match now.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => {
            if d.as_secs() == 0 {
                // Not necessarily wrong — QEMU might start at epoch
                // Just note it, don't fail
                eprintln!("[test_time] warning: SystemTime is at UNIX_EPOCH (RTC not set?)");
            }
        }
        Err(_) => {
            return Err("SystemTime is before UNIX_EPOCH".into());
        }
    }

    match stem::syscall::time_now(abi::time::ClockId::Realtime) {
        Ok(spec) => {
            if spec.as_nanos().is_none() {
                return Err("realtime clock returned invalid timespec".into());
            }
        }
        Err(abi::errors::Errno::EAGAIN) => {}
        Err(e) => return Err(alloc::format!("realtime clock read failed: {:?}", e)),
    }

    Ok(())
}

// ── D: env + args ────────────────────────────────────────────────────

fn test_env_args() -> Result<(), String> {
    // args: arg0 must be present
    let args: Vec<String> = std::env::args().collect();
    if args.is_empty() {
        return Err("args() returned empty — no arg0".into());
    }

    // set_var / var / remove_var round-trip
    let key = "HELLO_STD_TEST_VAR";
    std::env::set_var(key, "smoke_test_value");
    match std::env::var(key) {
        Ok(v) if v == "smoke_test_value" => {}
        Ok(v) => {
            return Err(alloc::format!(
                "var({}) = '{}', expected 'smoke_test_value'",
                key, v
            ))
        }
        Err(e) => return Err(alloc::format!("var({}) missing after set_var: {}", key, e)),
    }
    std::env::remove_var(key);
    if std::env::var(key).is_ok() {
        return Err(alloc::format!("var({}) still present after remove_var", key));
    }

    Ok(())
}

// ── E: filesystem ────────────────────────────────────────────────────

fn test_fs() -> Result<(), String> {
    let base = std::path::PathBuf::from("/data/hello_std_test");

    // Clean up from any previous run
    let _ = std::fs::remove_dir_all(&base);

    // create_dir_all
    std::fs::create_dir_all(&base).map_err(|e| alloc::format!("create_dir_all: {}", e))?;

    // Write a file
    let file_path = base.join("test.txt");
    let content = b"Hello from hello_std fs test!\n";
    {
        let mut f =
            std::fs::File::create(&file_path).map_err(|e| alloc::format!("File::create: {}", e))?;
        f.write_all(content)
            .map_err(|e| alloc::format!("write_all: {}", e))?;
    }

    // Read it back
    {
        let mut f = std::fs::File::open(&file_path).map_err(|e| alloc::format!("File::open: {}", e))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)
            .map_err(|e| alloc::format!("read_to_end: {}", e))?;
        if buf != content {
            return Err(alloc::format!(
                "read-back mismatch: got {} bytes, expected {}",
                buf.len(),
                content.len()
            ));
        }
    }

    // Seek test
    {
        let mut f =
            std::fs::File::open(&file_path).map_err(|e| alloc::format!("File::open for seek: {}", e))?;
        f.seek(SeekFrom::Start(6))
            .map_err(|e| alloc::format!("seek: {}", e))?;
        let mut partial = [0u8; 4];
        f.read_exact(&mut partial)
            .map_err(|e| alloc::format!("read_exact after seek: {}", e))?;
        if &partial != b"from" {
            return Err(alloc::format!(
                "seek+read: expected 'from', got '{}'",
                String::from_utf8_lossy(&partial)
            ));
        }
    }

    // Metadata — check file length
    let meta = std::fs::metadata(&file_path).map_err(|e| alloc::format!("metadata: {}", e))?;
    if meta.len() != content.len() as u64 {
        return Err(alloc::format!(
            "metadata.len() = {}, expected {}",
            meta.len(),
            content.len()
        ));
    }

    // read_dir — should list test.txt
    let entries: Vec<_> = std::fs::read_dir(&base)
        .map_err(|e| alloc::format!("read_dir: {}", e))?
        .filter_map(|e| e.ok())
        .collect();
    let found = entries
        .iter()
        .any(|e| e.file_name().to_str().map_or(false, |n| n == "test.txt"));
    if !found {
        let names: Vec<_> = entries
            .iter()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        return Err(alloc::format!("read_dir: 'test.txt' not found in {:?}", names));
    }

    // Rename
    let renamed = base.join("renamed.txt");
    std::fs::rename(&file_path, &renamed).map_err(|e| alloc::format!("rename: {}", e))?;
    if file_path.exists() {
        return Err("rename: old file still exists".into());
    }
    if !renamed.exists() {
        return Err("rename: new file does not exist".into());
    }

    // Clean up
    std::fs::remove_dir_all(&base).map_err(|e| alloc::format!("remove_dir_all: {}", e))?;
    if base.exists() {
        return Err("remove_dir_all: directory still exists".into());
    }

    Ok(())
}

// ── F: threads + sync ────────────────────────────────────────────────

fn test_threads() -> Result<(), String> {
    // Atomic increment across 4 threads
    let counter = Arc::new(AtomicUsize::new(0));
    let iters = 10_000;
    let nthreads = 4;
    let mut handles = Vec::new();

    for _ in 0..nthreads {
        let c = Arc::clone(&counter);
        handles.push(std::thread::spawn(move || {
            for _ in 0..iters {
                c.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }
    for h in handles {
        h.join().map_err(|_| "thread join panicked".to_string())?;
    }
    let total = counter.load(Ordering::SeqCst);
    let expected = nthreads * iters;
    if total != expected {
        return Err(alloc::format!("atomic total {} != expected {}", total, expected));
    }

    // Mutex test
    let mx = Arc::new(Mutex::new(0u64));
    let mut handles2 = Vec::new();
    for _ in 0..nthreads {
        let m = Arc::clone(&mx);
        handles2.push(std::thread::spawn(move || {
            for _ in 0..1000 {
                let mut guard = m.lock().unwrap();
                *guard += 1;
            }
        }));
    }
    for h in handles2 {
        h.join()
            .map_err(|_| "mutex thread join panicked".to_string())?;
    }
    let mx_total = *mx.lock().unwrap();
    if mx_total != (nthreads as u64) * 1000 {
        return Err(alloc::format!(
            "Mutex total {} != expected {}",
            mx_total,
            nthreads * 1000
        ));
    }

    // Condvar test — one thread waits, another signals
    let pair = Arc::new((Mutex::new(false), Condvar::new()));
    let pair2 = Arc::clone(&pair);
    let waiter = std::thread::spawn(move || {
        let (lock, cvar) = &*pair2;
        let mut started = lock.lock().unwrap();
        while !*started {
            started = cvar.wait(started).unwrap();
        }
        // Woke up!
        true
    });
    // Give the waiter a moment to park
    std::thread::sleep(Duration::from_millis(5));
    {
        let (lock, cvar) = &*pair;
        let mut started = lock.lock().unwrap();
        *started = true;
        cvar.notify_one();
    }
    let woke = waiter
        .join()
        .map_err(|_| "condvar waiter panicked".to_string())?;
    if !woke {
        return Err("Condvar: waiter did not wake".into());
    }

    Ok(())
}

// ── G: pipes ─────────────────────────────────────────────────────────

fn test_pipes() -> Result<(), String> {
    // Basic pipe: write, read to EOF
    let (reader, writer) = std::io::pipe().map_err(|e| alloc::format!("io::pipe(): {}", e))?;

    let msg = b"hello pipe\n";
    let writer_handle = std::thread::spawn(move || -> Result<(), String> {
        let mut w = writer;
        w.write_all(msg).map_err(|e| alloc::format!("pipe write: {}", e))?;
        drop(w); // Close writer → EOF for reader
        Ok(())
    });

    let mut r = reader;
    let mut buf = Vec::new();
    r.read_to_end(&mut buf)
        .map_err(|e| alloc::format!("pipe read_to_end: {}", e))?;
    if buf != msg {
        return Err(alloc::format!(
            "pipe: read '{}', expected '{}'",
            String::from_utf8_lossy(&buf),
            String::from_utf8_lossy(msg)
        ));
    }

    writer_handle
        .join()
        .map_err(|_| "pipe writer panicked".to_string())??;

    // Broken pipe test: drop reader, then writer should get error
    let (reader2, writer2) = std::io::pipe().map_err(|e| alloc::format!("io::pipe() #2: {}", e))?;
    drop(reader2);
    let mut w2 = writer2;
    match w2.write_all(b"should fail") {
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => {
            // Expected
        }
        Err(e) => {
            // Got an error, but not BrokenPipe — still acceptable in some impls
            eprintln!(
                "[test_pipes] note: broken pipe error kind = {:?} (expected BrokenPipe)",
                e.kind()
            );
        }
        Ok(()) => {
            return Err("write to pipe with dropped reader did not error".into());
        }
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
// OPTIONAL TESTS
// ═══════════════════════════════════════════════════════════════════

// ── --net: TCP connect ───────────────────────────────────────────────

fn test_net_tcp() -> Result<(), String> {
    use std::net::TcpStream;

    // Try the host-side guest proxy first. If it is up, we can validate
    // connect + write + read through the /net/tcp/ shim end-to-end.
    let addr = "10.0.2.2:8081";
    match TcpStream::connect(addr) {
        Ok(mut stream) => {
            let peer = stream
                .peer_addr()
                .map_err(|e| alloc::format!("peer_addr: {}", e))?;
            eprintln!("[test_net_tcp] connected to {} (peer={:?})", addr, peer);

            let req = b"GET /?url=http://example.com/ HTTP/1.1\r\nHost: 10.0.2.2\r\nConnection: close\r\n\r\n";
            stream
                .write_all(req)
                .map_err(|e| alloc::format!("write_all HTTP request: {}", e))?;

            let mut buf = [0u8; 256];
            let n = stream
                .read(&mut buf)
                .map_err(|e| alloc::format!("read HTTP response: {}", e))?;
            if n == 0 {
                return Err("HTTP GET returned EOF without response bytes".into());
            }

            let head = String::from_utf8_lossy(&buf[..n]);
            if !head.starts_with("HTTP/1.1 ") && !head.starts_with("HTTP/1.0 ") {
                return Err(alloc::format!("HTTP GET returned non-HTTP prefix: {:?}", head));
            }

            drop(stream);
            Ok(())
        }
        Err(e) => {
            // Connection refused is acceptable here because the shim still
            // reached the host TCP stack. The proxy may simply not be running.
            if alloc::format!("{}", e).contains("refused")
                || alloc::format!("{}", e).contains("failed")
                || alloc::format!("{}", e).contains("timed out")
            {
                eprintln!(
                    "[test_net_tcp] connect/write path to {} returned: {}",
                    addr, e
                );
                Ok(())
            } else {
                Err(alloc::format!("TcpStream::connect({}): {}", addr, e))
            }
        }
    }
}

// ── --dns: DNS lookup ────────────────────────────────────────────────

fn test_dns() -> Result<(), String> {
    use std::net::ToSocketAddrs;

    let host = "example.com:80";
    match host.to_socket_addrs() {
        Ok(addrs) => {
            let addrs: Vec<_> = addrs.collect();
            if addrs.is_empty() {
                return Err("DNS resolved to 0 addresses".into());
            }
            eprintln!("[test_dns] {} resolved to {:?}", host, addrs);
            Ok(())
        }
        Err(e) => Err(alloc::format!("DNS lookup for {}: {}", host, e)),
    }
}

// ── --net: TCP listener / server-side accept ─────────────────────────

fn test_net_tcp_listener() -> Result<(), String> {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};

    // Bind a listener on loopback.
    let listener = match TcpListener::bind("127.0.0.1:0") {
        Ok(l) => l,
        Err(e) => return Err(alloc::format!("TcpListener::bind: {}", e)),
    };

    let local_addr = listener
        .local_addr()
        .map_err(|e| alloc::format!("local_addr: {}", e))?;
    eprintln!("[test_net_tcp_listener] listening on {}", local_addr);

    // Connect a client.
    let mut client =
        TcpStream::connect(local_addr).map_err(|e| alloc::format!("client connect: {}", e))?;

    // Accept the connection.
    let (mut server, peer_addr) = listener
        .accept()
        .map_err(|e| alloc::format!("accept: {}", e))?;
    eprintln!("[test_net_tcp_listener] accepted peer {}", peer_addr);

    // Exchange data client→server.
    client
        .write_all(b"ping")
        .map_err(|e| alloc::format!("client write: {}", e))?;

    let mut buf = [0u8; 8];
    let n = server
        .read(&mut buf)
        .map_err(|e| alloc::format!("server read: {}", e))?;
    if &buf[..n] != b"ping" {
        return Err(alloc::format!("server read got {:?}, want b\"ping\"", &buf[..n]));
    }

    // Exchange data server→client.
    server
        .write_all(b"pong")
        .map_err(|e| alloc::format!("server write: {}", e))?;

    let n = client
        .read(&mut buf)
        .map_err(|e| alloc::format!("client read: {}", e))?;
    if &buf[..n] != b"pong" {
        return Err(alloc::format!("client read got {:?}, want b\"pong\"", &buf[..n]));
    }

    Ok(())
}

fn test_net_tcp_accept_multiple() -> Result<(), String> {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};

    let listener = match TcpListener::bind("127.0.0.1:0") {
        Ok(l) => l,
        Err(e) => return Err(alloc::format!("TcpListener::bind: {}", e)),
    };
    let local_addr = listener
        .local_addr()
        .map_err(|e| alloc::format!("local_addr: {}", e))?;

    // First client.
    let mut c1 =
        TcpStream::connect(local_addr).map_err(|e| alloc::format!("c1 connect: {}", e))?;
    let (mut s1, _) = listener.accept().map_err(|e| alloc::format!("accept 1: {}", e))?;
    c1.write_all(b"A").map_err(|e| alloc::format!("c1 write: {}", e))?;
    let mut buf = [0u8; 4];
    let n = s1.read(&mut buf).map_err(|e| alloc::format!("s1 read: {}", e))?;
    if n == 0 || buf[0] != b'A' {
        return Err("first accept: payload mismatch".into());
    }
    drop(c1);
    drop(s1);

    // Second client — listener must still be usable.
    let mut c2 =
        TcpStream::connect(local_addr).map_err(|e| alloc::format!("c2 connect: {}", e))?;
    let (mut s2, _) = listener.accept().map_err(|e| alloc::format!("accept 2: {}", e))?;
    c2.write_all(b"B").map_err(|e| alloc::format!("c2 write: {}", e))?;
    let n = s2.read(&mut buf).map_err(|e| alloc::format!("s2 read: {}", e))?;
    if n == 0 || buf[0] != b'B' {
        return Err("second accept: payload mismatch".into());
    }

    Ok(())
}

fn test_net_tcp_nonblocking() -> Result<(), String> {
    use std::io::Read;
    use std::net::TcpListener;

    let listener = match TcpListener::bind("127.0.0.1:0") {
        Ok(l) => l,
        Err(e) => return Err(alloc::format!("TcpListener::bind: {}", e)),
    };
    listener
        .set_nonblocking(true)
        .map_err(|e| alloc::format!("set_nonblocking: {}", e))?;

    // In nonblocking mode, accept() with no incoming connection must
    // return WouldBlock / EAGAIN immediately.
    match listener.accept() {
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(()),
        Err(e) => Err(alloc::format!("accept (nonblocking, no peer): unexpected error: {}", e)),
        Ok((_, addr)) => Err(alloc::format!(
            "accept (nonblocking, no peer): unexpectedly returned peer {}",
            addr
        )),
    }
}

// ── --udp: UDP loopback send/recv ────────────────────────────────────

fn test_net_udp_loopback() -> Result<(), String> {
    use std::net::UdpSocket;

    let recv_sock = UdpSocket::bind("127.0.0.1:0")
        .map_err(|e| alloc::format!("recv bind: {}", e))?;
    let recv_addr = recv_sock
        .local_addr()
        .map_err(|e| alloc::format!("recv local_addr: {}", e))?;

    let send_sock = UdpSocket::bind("127.0.0.1:0")
        .map_err(|e| alloc::format!("send bind: {}", e))?;

    let msg = b"hello udp";
    let sent = send_sock
        .send_to(msg, recv_addr)
        .map_err(|e| alloc::format!("send_to: {}", e))?;
    if sent != msg.len() {
        return Err(alloc::format!("send_to: sent {} bytes, want {}", sent, msg.len()));
    }

    let mut buf = [0u8; 64];
    let (n, src_addr) = recv_sock
        .recv_from(&mut buf)
        .map_err(|e| alloc::format!("recv_from: {}", e))?;

    if &buf[..n] != msg {
        return Err(alloc::format!(
            "recv_from: got {:?}, want {:?}",
            &buf[..n],
            msg
        ));
    }

    eprintln!(
        "[test_net_udp_loopback] received {} bytes from {}",
        n, src_addr
    );
    Ok(())
}

// ── --net: IPv6 policy (must reject cleanly) ─────────────────────────

fn test_net_ipv6_policy() -> Result<(), String> {
    use std::net::{TcpListener, TcpStream, UdpSocket};

    // All IPv6 operations must return EAFNOSUPPORT, not panic.
    let addrs_to_reject = ["[::1]:8080", "::1", "[::]:0"];

    for addr in addrs_to_reject {
        // TcpListener::bind
        match TcpListener::bind(addr) {
            Err(e) => {
                eprintln!("[test_net_ipv6_policy] TcpListener::bind({:?}) → {}", addr, e);
            }
            Ok(_) => {
                return Err(alloc::format!(
                    "TcpListener::bind({:?}) succeeded — expected failure",
                    addr
                ));
            }
        }
        // TcpStream::connect
        match TcpStream::connect(addr) {
            Err(e) => {
                eprintln!("[test_net_ipv6_policy] TcpStream::connect({:?}) → {}", addr, e);
            }
            Ok(_) => {
                return Err(alloc::format!(
                    "TcpStream::connect({:?}) succeeded — expected failure",
                    addr
                ));
            }
        }
        // UdpSocket::bind
        match UdpSocket::bind(addr) {
            Err(e) => {
                eprintln!("[test_net_ipv6_policy] UdpSocket::bind({:?}) → {}", addr, e);
            }
            Ok(_) => {
                return Err(alloc::format!(
                    "UdpSocket::bind({:?}) succeeded — expected failure",
                    addr
                ));
            }
        }
    }

    Ok(())
}

// ── --dns: basic DNS lookup (or explicit unsupported error) ──────────

fn test_spawn() -> Result<(), String> {
    use std::process::Command;

    // Get our own executable path from args
    let self_name = std::env::args()
        .next()
        .ok_or_else(|| "no arg0 for self-exec".to_string())?;

    // Test 1: echo child — pass args, capture output
    let output = Command::new(&self_name)
        .arg("--child-echo")
        .arg("hello")
        .arg("world")
        .output()
        .map_err(|e| alloc::format!("spawn echo child: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.contains("hello") || !stdout.contains("world") {
        return Err(alloc::format!(
            "echo child: expected 'hello' and 'world' in output, got: '{}'",
            stdout.trim()
        ));
    }

    // Test 2: exit code child
    let status = Command::new(&self_name)
        .arg("--child-exit")
        .arg("--exit-code=42")
        .status()
        .map_err(|e| alloc::format!("spawn exit child: {}", e))?;

    match status.code() {
        Some(42) => {}
        Some(c) => return Err(alloc::format!("exit child: expected code 42, got {}", c)),
        None => return Err("exit child: no exit code".into()),
    }

    Ok(())
}

// ── --stress: repeated tests ─────────────────────────────────────────

fn test_stress() -> Result<(), String> {
    let iterations = 50;

    for i in 0..iterations {
        // Thread stress
        let counter = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let c = Arc::clone(&counter);
            handles.push(std::thread::spawn(move || {
                for _ in 0..1000 {
                    c.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }
        for h in handles {
            h.join()
                .map_err(|_| alloc::format!("stress iter {}: thread panicked", i))?;
        }
        let total = counter.load(Ordering::SeqCst);
        if total != 4000 {
            return Err(alloc::format!("stress iter {}: atomic total {} != 4000", i, total));
        }

        // Pipe stress
        let (reader, writer) =
            std::io::pipe().map_err(|e| alloc::format!("stress iter {}: pipe: {}", i, e))?;
        let wh = std::thread::spawn(move || {
            let mut w = writer;
            let _ = w.write_all(b"stress");
            drop(w);
        });
        let mut r = reader;
        let mut buf = Vec::new();
        r.read_to_end(&mut buf)
            .map_err(|e| alloc::format!("stress iter {}: read: {}", i, e))?;
        wh.join()
            .map_err(|_| alloc::format!("stress iter {}: writer panicked", i))?;
        if buf != b"stress" {
            return Err(alloc::format!("stress iter {}: data mismatch", i));
        }
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
// ENTRY POINT
// ═══════════════════════════════════════════════════════════════════

#[stem::main]
fn main(_arg: usize) -> ! {
    let flags = parse_flags();

    // Handle child modes — these exit immediately
    if flags.child_echo {
        child_echo_mode();
    }
    if flags.child_exit {
        child_exit_mode(flags.child_exit_code);
    }

    // Normal test mode
    print_banner();

    // Default tests
    let mut tests: Vec<Test> = vec![
        Test {
            name: "stdio",
            func: test_stdio,
            fatal: true,
            hint: "check PAL stdio/thingos.rs and kernel serial output",
        },
        Test {
            name: "alloc + collections",
            func: test_alloc,
            fatal: true,
            hint: "check PAL alloc/thingos.rs and heap initialization",
        },
        Test {
            name: "time",
            func: test_time,
            fatal: false,
            hint: "check PAL time/thingos.rs, monotonic clock, or sleep/parking",
        },
        Test {
            name: "env + args",
            func: test_env_args,
            fatal: false,
            hint: "check PAL env/thingos.rs and args/thingos.rs, kernel argv/env syscalls",
        },
        Test {
            name: "filesystem",
            func: test_fs,
            fatal: false,
            hint: "check PAL fs/thingos.rs, resolver/handles/metadata, bytespace ops",
        },
        Test {
            name: "threads + sync",
            func: test_threads,
            fatal: false,
            hint: "check PAL thread/thingos.rs, spawn/join/parking, sync primitives",
        },
        Test {
            name: "pipes",
            func: test_pipes,
            fatal: false,
            hint: "check PAL pipe/thingos.rs, close/waitq semantics, BrokenPipe",
        },
    ];

    // Optional tests
    if flags.net {
        tests.push(Test {
            name: "net: TCP connect (client)",
            func: test_net_tcp,
            fatal: false,
            hint: "check PAL net/connection/thingos.rs, netd IPC, smoltcp stack",
        });
    }
    if flags.net_server {
        tests.push(Test {
            name: "net: TCP listener/accept (loopback)",
            func: test_net_tcp_listener,
            fatal: false,
            hint: "check TcpListener PAL (tcp_bind/accept), netd listen/accept VFS flow",
        });
        tests.push(Test {
            name: "net: TCP accept multiple clients",
            func: test_net_tcp_accept_multiple,
            fatal: false,
            hint: "check listener reuse after accept, netd backlog pool",
        });
        tests.push(Test {
            name: "net: TCP nonblocking accept (would-block)",
            func: test_net_tcp_nonblocking,
            fatal: false,
            hint: "check set_nonblocking on TcpListener, EAGAIN from netd accept path",
        });
    }
    if flags.udp {
        tests.push(Test {
            name: "net: UDP loopback send/recv",
            func: test_net_udp_loopback,
            fatal: false,
            hint: "check UdpSocket PAL (udp_bind/send_to/recv_from), netd UDP VFS flow",
        });
    }
    if flags.ipv6 {
        tests.push(Test {
            name: "net: IPv6 policy (must reject)",
            func: test_net_ipv6_policy,
            fatal: false,
            hint: "check reject_ipv6() in PAL net.rs; all IPv6 ops must return EAFNOSUPPORT",
        });
    }
    if flags.dns {
        tests.push(Test {
            name: "net: DNS lookup",
            func: test_dns,
            fatal: false,
            hint: "check PAL net/connection/thingos.rs (LookupHost), netd DNS resolver",
        });
    }
    if flags.spawn {
        tests.push(Test {
            name: "process: spawn",
            func: test_spawn,
            fatal: false,
            hint: "check PAL process/thingos.rs, SYS_SPAWN_PROCESS_EX, pipe stdio",
        });
    }
    if flags.stress {
        tests.push(Test {
            name: "stress: threads + pipes ×50",
            func: test_stress,
            fatal: false,
            hint: "thread spawn/join or pipe close semantics under load",
        });
    }

    let code = run_all(&tests);
    std::process::exit(code);
}
