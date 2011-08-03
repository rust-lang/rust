use std;
import std::run;
import std::os;
import std::io;
import std::option;
import std::str;

// Regression test for memory leaks
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[test]
fn test_leaks() {
    run::run_program("echo", []);
    run::start_program("echo", []);
    run::program_output("echo", []);
}

// FIXME
#[cfg(target_os = "win32")]
#[test]
#[ignore]
fn test_leaks() { }

#[test]
fn test_pipes() {
    let pipe_in = os::pipe();
    let pipe_out = os::pipe();
    let pipe_err = os::pipe();

    let pid = run::spawn_process("cat", [],
       pipe_in.in, pipe_out.out, pipe_err.out);
    os::libc::close(pipe_in.in);
    os::libc::close(pipe_out.out);
    os::libc::close(pipe_err.out);

    if pid == -1 { fail; }
    let expected = "test";
    writeclose(pipe_in.out, expected);
    let actual = readclose(pipe_out.in);
    readclose(pipe_err.in);
    os::waitpid(pid);

    log expected;
    log actual;
    assert expected == actual;

    fn writeclose(fd: int, s: &str) {
        let writer = io::new_writer(
            io::fd_buf_writer(fd, option::none));
        writer.write_str(s);

        os::libc::close(fd);
    }

    fn readclose(fd: int) -> str {
        // Copied from run::program_output
        let file = os::fd_FILE(fd);
        let reader = io::new_reader(io::FILE_buf_reader(file, option::none));
        let buf = "";
        while !reader.eof() {
            let bytes = reader.read_bytes(4096u);
            buf += str::unsafe_from_bytes(bytes);
        }
        os::libc::fclose(file);
        ret buf;
    }
}