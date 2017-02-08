// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// This is a small client program intended to pair with `qemu-test-server` in
/// this repository. This client connects to the server over TCP and is used to
/// push artifacts and run tests on the server instead of locally.
///
/// Here is also where we bake in the support to spawn the QEMU emulator as
/// well.

use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::io::{self, BufWriter};
use std::net::TcpStream;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

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
            spawn_emulator(Path::new(&args.next().unwrap()),
                           Path::new(&args.next().unwrap()))
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

fn spawn_emulator(rootfs: &Path, tmpdir: &Path) {
    // Generate a new rootfs image now that we've updated the test server
    // executable. This is the equivalent of:
    //
    //      find $rootfs -print 0 | cpio --null -o --format=newc > rootfs.img
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
    let mut cmd = Command::new("qemu-system-arm");
    cmd.arg("-M").arg("vexpress-a15")
       .arg("-m").arg("1024")
       .arg("-kernel").arg("/tmp/zImage")
       .arg("-initrd").arg(&rootfs_img)
       .arg("-dtb").arg("/tmp/vexpress-v2p-ca15-tc1.dtb")
       .arg("-append").arg("console=ttyAMA0 root=/dev/ram rdinit=/sbin/init init=/sbin/init")
       .arg("-nographic")
       .arg("-redir").arg("tcp:12345::12345");
    t!(cmd.spawn());

    // Wait for the emulator to come online
    loop {
        let dur = Duration::from_millis(100);
        if let Ok(mut client) = TcpStream::connect("127.0.0.1:12345") {
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

    fn add_files(w: &mut Write, root: &Path, cur: &Path) {
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
    let client = t!(TcpStream::connect("127.0.0.1:12345"));
    let mut client = BufWriter::new(client);
    t!(client.write_all(b"push"));
    t!(client.write_all(path.file_name().unwrap().to_str().unwrap().as_bytes()));
    t!(client.write_all(&[0]));
    let mut file = t!(File::open(path));
    t!(io::copy(&mut file, &mut client));
    t!(client.flush());
    println!("done pushing {:?}", path);
}

fn run(files: String, args: Vec<String>) {
    let client = t!(TcpStream::connect("127.0.0.1:12345"));
    let mut client = BufWriter::new(client);
    t!(client.write_all(b"run "));

    // Send over the args
    for arg in args {
        t!(client.write_all(arg.as_bytes()));
        t!(client.write_all(&[0]));
    }
    t!(client.write_all(&[0]));

    // Send over env vars
    for (k, v) in env::vars() {
        if k != "PATH" && k != "LD_LIBRARY_PATH" {
            t!(client.write_all(k.as_bytes()));
            t!(client.write_all(&[0]));
            t!(client.write_all(v.as_bytes()));
            t!(client.write_all(&[0]));
        }
    }
    t!(client.write_all(&[0]));

    // Send over support libraries
    let mut files = files.split(':');
    let exe = files.next().unwrap();
    for file in files.map(Path::new) {
        t!(client.write_all(file.file_name().unwrap().to_str().unwrap().as_bytes()));
        t!(client.write_all(&[0]));
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

fn send(path: &Path, dst: &mut Write) {
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
