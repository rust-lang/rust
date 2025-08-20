#![feature(command_pass_fds)]
#![cfg(unix)]

use std::io::{self, Write};
use std::os::unix::io::AsRawFd;
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};

use libc::{F_GETFD, fcntl};

#[test]
fn fd_test_stdin() {
    let (pipe_reader, mut pipe_writer) = io::pipe().unwrap();

    let fd_num = 0;

    let mut cmd = Command::new("cat");
    cmd.stdout(Stdio::piped()).fd(fd_num, pipe_reader);

    let mut child = cmd.spawn().unwrap();
    let mut stdout = child.stdout.take().unwrap();

    pipe_writer.write_all(b"Hello, world!").unwrap();
    drop(pipe_writer);

    child.wait().unwrap();
    assert_eq!(io::read_to_string(&mut stdout).unwrap(), "Hello, world!");
}

#[test]
fn fd_test_swap() {
    let (pipe_reader1, mut pipe_writer1) = io::pipe().unwrap();
    let (pipe_reader2, mut pipe_writer2) = io::pipe().unwrap();

    let num1 = pipe_reader1.as_raw_fd();
    let num2 = pipe_reader2.as_raw_fd();

    let mut cmd = Command::new("cat");
    cmd.arg(format!("/dev/fd/{}", num1))
        .stdout(Stdio::piped())
        .fd(num2, pipe_reader1)
        .fd(num1, pipe_reader2);

    let mut child = cmd.spawn().unwrap();
    let mut stdout = child.stdout.take().unwrap();

    pipe_writer1.write_all(b"Hello from pipe 1!").unwrap();
    drop(pipe_writer1);

    pipe_writer2.write_all(b"Hello from pipe 2!").unwrap();
    drop(pipe_writer2);

    child.wait().unwrap();
    assert_eq!(io::read_to_string(&mut stdout).unwrap(), "Hello from pipe 1!");
}

#[test]
fn fd_test_close_time() {
    let (pipe_reader, mut pipe_writer) = io::pipe().unwrap();

    let fd_num = 123;

    let original = pipe_reader.as_raw_fd();

    let mut cmd = Command::new("cat");
    cmd.arg(format!("/dev/fd/{fd_num}")).stdout(Stdio::piped()).fd(fd_num, pipe_reader);

    assert_ne!(unsafe { fcntl(original, F_GETFD) }, -1);

    let mut child = cmd.spawn().unwrap();
    let mut stdout = child.stdout.take().unwrap();

    pipe_writer.write_all(b"Hello, world!").unwrap();
    drop(pipe_writer);

    child.wait().unwrap();
    assert_eq!(io::read_to_string(&mut stdout).unwrap(), "Hello, world!");
    assert_eq!(unsafe { fcntl(original, F_GETFD) }, -1);
}
