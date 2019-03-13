#![deny(rust_2018_idioms)]

/// This is a small client program intended to pair with `remote-test-server` in
/// this repository. This client connects to the server over TCP and is used to
/// push artifacts and run tests on the server instead of locally.
///
/// Here is also where we bake in the support to spawn the QEMU emulator as
/// well.

use std::env;
use std::fs::{self, File};
use std::io::prelude::*;
use std::io::{self, BufWriter};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

const REMOTE_ADDR_ENV: &str = "TEST_DEVICE_ADDR";

macro_rules! t {
    ($e:expr) => (match $e {
        Ok(e) => e,
        Err(e) => panic!("{} failed with {}", stringify!($e), e),
    })
}

fn main() {
    let mut args = env::args().skip(1);

    match &args.next().unwrap()[..] {
        "spawn-emulator" => {
            spawn_emulator(&args.next().unwrap(),
                           Path::new(&args.next().unwrap()),
                           Path::new(&args.next().unwrap()),
                           args.next().map(|s| s.into()))
        }
        "push" => {
            push(Path::new(&args.next().unwrap()))
        }
        "run" => {
            run(args.next().unwrap(), args.collect())
        }
        cmd => panic!("unknown command: {}", cmd),
    }
}

fn spawn_emulator(target: &str,
                  server: &Path,
                  tmpdir: &Path,
                  rootfs: Option<PathBuf>) {
    let device_address = env::var(REMOTE_ADDR_ENV).unwrap_or("127.0.0.1:12345".to_string());

    if env::var(REMOTE_ADDR_ENV).is_ok() {
        println!("Connecting to remote device {} ...", device_address);
    } else if target.contains("android") {
        start_android_emulator(server);
    } else {
        let rootfs = rootfs.as_ref().expect("need rootfs on non-android");
        start_qemu_emulator(target, rootfs, server, tmpdir);
    }

    // Wait for the emulator to come online
    loop {
        let dur = Duration::from_millis(100);
        if let Ok(mut client) = TcpStream::connect(&device_address) {
            t!(client.set_read_timeout(Some(dur)));
            t!(client.set_write_timeout(Some(dur)));
            if client.write_all(b"ping").is_ok() {
                let mut b = [0; 4];
                if client.read_exact(&mut b).is_ok() {
                    break
                }
            }
        }
        thread::sleep(dur);
    }
}

fn start_android_emulator(server: &Path) {
    println!("waiting for device to come online");
    let status = Command::new("adb")
                    .arg("wait-for-device")
                    .status()
                    .unwrap();
    assert!(status.success());

    println!("pushing server");
    let status = Command::new("adb")
                    .arg("push")
                    .arg(server)
                    .arg("/data/tmp/testd")
                    .status()
                    .unwrap();
    assert!(status.success());

    println!("forwarding tcp");
    let status = Command::new("adb")
                    .arg("forward")
                    .arg("tcp:12345")
                    .arg("tcp:12345")
                    .status()
                    .unwrap();
    assert!(status.success());

    println!("executing server");
    Command::new("adb")
                    .arg("shell")
                    .arg("/data/tmp/testd")
                    .spawn()
                    .unwrap();
}

fn start_qemu_emulator(target: &str,
                       rootfs: &Path,
                       server: &Path,
                       tmpdir: &Path) {
    // Generate a new rootfs image now that we've updated the test server
    // executable. This is the equivalent of:
    //
    //      find $rootfs -print 0 | cpio --null -o --format=newc > rootfs.img
    t!(fs::copy(server, rootfs.join("testd")));
    let rootfs_img = tmpdir.join("rootfs.img");
    let mut cmd = Command::new("cpio");
    cmd.arg("--null")
       .arg("-o")
       .arg("--format=newc")
       .stdin(Stdio::piped())
       .stdout(Stdio::piped())
       .current_dir(rootfs);
    let mut child = t!(cmd.spawn());
    let mut stdin = child.stdin.take().unwrap();
    let rootfs = rootfs.to_path_buf();
    thread::spawn(move || add_files(&mut stdin, &rootfs, &rootfs));
    t!(io::copy(&mut child.stdout.take().unwrap(),
                &mut t!(File::create(&rootfs_img))));
    assert!(t!(child.wait()).success());

    // Start up the emulator, in the background
    match target {
        "arm-unknown-linux-gnueabihf" => {
            let mut cmd = Command::new("qemu-system-arm");
            cmd.arg("-M").arg("vexpress-a15")
               .arg("-m").arg("1024")
               .arg("-kernel").arg("/tmp/zImage")
               .arg("-initrd").arg(&rootfs_img)
               .arg("-dtb").arg("/tmp/vexpress-v2p-ca15-tc1.dtb")
               .arg("-append")
               .arg("console=ttyAMA0 root=/dev/ram rdinit=/sbin/init init=/sbin/init")
               .arg("-nographic")
               .arg("-redir").arg("tcp:12345::12345");
            t!(cmd.spawn());
        }
        "aarch64-unknown-linux-gnu" => {
            let mut cmd = Command::new("qemu-system-aarch64");
            cmd.arg("-machine").arg("virt")
               .arg("-cpu").arg("cortex-a57")
               .arg("-m").arg("1024")
               .arg("-kernel").arg("/tmp/Image")
               .arg("-initrd").arg(&rootfs_img)
               .arg("-append")
               .arg("console=ttyAMA0 root=/dev/ram rdinit=/sbin/init init=/sbin/init")
               .arg("-nographic")
               .arg("-netdev").arg("user,id=net0,hostfwd=tcp::12345-:12345")
               .arg("-device").arg("virtio-net-device,netdev=net0,mac=00:00:00:00:00:00");
            t!(cmd.spawn());
        }
        _ => panic!("cannot start emulator for: {}"< target),
    }

    fn add_files(w: &mut dyn Write, root: &Path, cur: &Path) {
        for entry in t!(cur.read_dir()) {
            let entry = t!(entry);
            let path = entry.path();
            let to_print = path.strip_prefix(root).unwrap();
            t!(write!(w, "{}\u{0}", to_print.to_str().unwrap()));
            if t!(entry.file_type()).is_dir() {
                add_files(w, root, &path);
            }
        }
    }
}

fn push(path: &Path) {
    let device_address = env::var(REMOTE_ADDR_ENV).unwrap_or("127.0.0.1:12345".to_string());
    let client = t!(TcpStream::connect(device_address));
    let mut client = BufWriter::new(client);
    t!(client.write_all(b"push"));
    send(path, &mut client);
    t!(client.flush());

    // Wait for an acknowledgement that all the data was received. No idea
    // why this is necessary, seems like it shouldn't be!
    let mut client = client.into_inner().unwrap();
    let mut buf = [0; 4];
    t!(client.read_exact(&mut buf));
    assert_eq!(&buf, b"ack ");
    println!("done pushing {:?}", path);
}

fn run(files: String, args: Vec<String>) {
    let device_address = env::var(REMOTE_ADDR_ENV).unwrap_or("127.0.0.1:12345".to_string());
    let client = t!(TcpStream::connect(device_address));
    let mut client = BufWriter::new(client);
    t!(client.write_all(b"run "));

    // Send over the args
    for arg in args {
        t!(client.write_all(arg.as_bytes()));
        t!(client.write_all(&[0]));
    }
    t!(client.write_all(&[0]));

    // Send over env vars
    //
    // Don't send over *everything* though as some env vars are set by and used
    // by the client.
    for (k, v) in env::vars() {
        match &k[..] {
            "PATH" |
            "LD_LIBRARY_PATH" |
            "PWD" => continue,
            _ => {}
        }
        t!(client.write_all(k.as_bytes()));
        t!(client.write_all(&[0]));
        t!(client.write_all(v.as_bytes()));
        t!(client.write_all(&[0]));
    }
    t!(client.write_all(&[0]));

    // Send over support libraries
    let mut files = files.split(':');
    let exe = files.next().unwrap();
    for file in files.map(Path::new) {
        send(&file, &mut client);
    }
    t!(client.write_all(&[0]));

    // Send over the client executable as the last piece
    send(exe.as_ref(), &mut client);

    println!("uploaded {:?}, waiting for result", exe);

    // Ok now it's time to read all the output. We're receiving "frames"
    // representing stdout/stderr, so we decode all that here.
    let mut header = [0; 5];
    let mut stderr_done = false;
    let mut stdout_done = false;
    let mut client = t!(client.into_inner());
    let mut stdout = io::stdout();
    let mut stderr = io::stderr();
    while !stdout_done || !stderr_done {
        t!(client.read_exact(&mut header));
        let amt = ((header[1] as u64) << 24) |
                  ((header[2] as u64) << 16) |
                  ((header[3] as u64) <<  8) |
                  ((header[4] as u64) <<  0);
        if header[0] == 0 {
            if amt == 0 {
                stdout_done = true;
            } else {
                t!(io::copy(&mut (&mut client).take(amt), &mut stdout));
                t!(stdout.flush());
            }
        } else {
            if amt == 0 {
                stderr_done = true;
            } else {
                t!(io::copy(&mut (&mut client).take(amt), &mut stderr));
                t!(stderr.flush());
            }
        }
    }

    // Finally, read out the exit status
    let mut status = [0; 5];
    t!(client.read_exact(&mut status));
    let code = ((status[1] as i32) << 24) |
               ((status[2] as i32) << 16) |
               ((status[3] as i32) <<  8) |
               ((status[4] as i32) <<  0);
    if status[0] == 0 {
        std::process::exit(code);
    } else {
        println!("died due to signal {}", code);
        std::process::exit(3);
    }
}

fn send(path: &Path, dst: &mut dyn Write) {
    t!(dst.write_all(path.file_name().unwrap().to_str().unwrap().as_bytes()));
    t!(dst.write_all(&[0]));
    let mut file = t!(File::open(&path));
    let amt = t!(file.metadata()).len();
    t!(dst.write_all(&[
        (amt >> 24) as u8,
        (amt >> 16) as u8,
        (amt >>  8) as u8,
        (amt >>  0) as u8,
    ]));
    t!(io::copy(&mut file, dst));
}
