import core::*;

use std;
import std::run;
import std::os;
import std::io;
import option;
import str;
import ctypes::fd_t;

// Regression test for memory leaks
#[ignore(cfg(target_os = "win32"))] // FIXME
fn test_leaks() {
    run::run_program("echo", []);
    run::start_program("echo", []);
    run::program_output("echo", []);
}

#[test]
fn test_pipes() {
    let pipe_in = os::pipe();
    let pipe_out = os::pipe();
    let pipe_err = os::pipe();

    let pid =
        run::spawn_process("cat", [], pipe_in.in, pipe_out.out, pipe_err.out);
    os::close(pipe_in.in);
    os::close(pipe_out.out);
    os::close(pipe_err.out);

    if pid == -1i32 { fail; }
    let expected = "test";
    writeclose(pipe_in.out, expected);
    let actual = readclose(pipe_out.in);
    readclose(pipe_err.in);
    os::waitpid(pid);

    log_full(core::debug, expected);
    log_full(core::debug, actual);
    assert (expected == actual);

    fn writeclose(fd: fd_t, s: str) {
        log_err("writeclose", (fd, s));
        let writer = io::new_writer(io::fd_buf_writer(fd, option::none));
        writer.write_str(s);

        os::close(fd);
    }

    fn readclose(fd: fd_t) -> str {
        // Copied from run::program_output
        let file = os::fd_FILE(fd);
        let reader = io::new_reader(io::FILE_buf_reader(file, option::none));
        let buf = "";
        while !reader.eof() {
            let bytes = reader.read_bytes(4096u);
            buf += str::unsafe_from_bytes(bytes);
        }
        os::fclose(file);
        ret buf;
    }
}

#[test]
fn waitpid() {
    let pid = run::spawn_process("false", [], 0i32, 0i32, 0i32);
    let status = run::waitpid(pid);
    assert status == 1;
}
