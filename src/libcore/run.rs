#[doc ="Process spawning"];
import option::{some, none};
import libc::{pid_t, c_void, c_int};

export program;
export run_program;
export start_program;
export program_output;
export spawn_process;
export waitpid;

#[abi = "cdecl"]
native mod rustrt {
    fn rust_run_program(argv: **u8, envp: *c_void, dir: *u8,
                        in_fd: c_int, out_fd: c_int, err_fd: c_int)
        -> pid_t;
}

#[doc ="A value representing a child process"]
iface program {
    #[doc ="Returns the process id of the program"]
    fn get_id() -> pid_t;

    #[doc ="Returns an io::writer that can be used to write to stdin"]
    fn input() -> io::writer;

    #[doc ="Returns an io::reader that can be used to read from stdout"]
    fn output() -> io::reader;

    #[doc ="Returns an io::reader that can be used to read from stderr"]
    fn err() -> io::reader;

    #[doc = "Closes the handle to the child processes standard input"]
    fn close_input();

    #[doc = "
    Waits for the child process to terminate. Closes the handle
    to stdin if necessary.
    "]
    fn finish() -> int;

    #[doc ="Closes open handles"]
    fn destroy();
}


#[doc = "
Run a program, providing stdin, stdout and stderr handles

# Arguments

* prog - The path to an executable
* args - Vector of arguments to pass to the child process
* env - optional env-modification for child
* dir - optional dir to run child in (default current dir)
* in_fd - A file descriptor for the child to use as std input
* out_fd - A file descriptor for the child to use as std output
* err_fd - A file descriptor for the child to use as std error

# Return value

The process id of the spawned process
"]
fn spawn_process(prog: str, args: [str],
                 env: option<[(str,str)]>,
                 dir: option<str>,
                 in_fd: c_int, out_fd: c_int, err_fd: c_int)
   -> pid_t unsafe {
    with_argv(prog, args) {|argv|
        with_envp(env) { |envp|
            with_dirp(dir) { |dirp|
                rustrt::rust_run_program(argv, envp, dirp,
                                         in_fd, out_fd, err_fd)
            }
        }
    }
}

fn with_argv<T>(prog: str, args: [str],
                cb: fn(**u8) -> T) -> T unsafe {
    let mut argptrs = str::as_buf(prog) {|b| [b] };
    let mut tmps = [];
    for arg in args {
        let t = @arg;
        tmps += [t];
        argptrs += str::as_buf(*t) {|b| [b] };
    }
    argptrs += [ptr::null()];
    vec::as_buf(argptrs, cb)
}

#[cfg(target_os = "macos")]
#[cfg(target_os = "linux")]
#[cfg(target_os = "freebsd")]
fn with_envp<T>(env: option<[(str,str)]>,
                cb: fn(*c_void) -> T) -> T unsafe {
    // On posixy systems we can pass a char** for envp, which is
    // a null-terminated array of "k=v\n" strings.
    alt env {
      some (es) {
        let mut tmps = [];
        let mut ptrs = [];

        for (k,v) in es {
            let t = @(#fmt("%s=%s", k, v));
            vec::push(tmps, t);
            ptrs += str::as_buf(*t) {|b| [b]};
        }
        ptrs += [ptr::null()];
        vec::as_buf(ptrs) { |p| cb(::unsafe::reinterpret_cast(p)) }
      }
      none {
        cb(ptr::null())
      }
    }
}

#[cfg(target_os = "win32")]
fn with_envp<T>(env: option<[(str,str)]>,
                cb: fn(*c_void) -> T) -> T unsafe {
    // On win32 we pass an "environment block" which is not a char**, but
    // rather a concatenation of null-terminated k=v\0 sequences, with a final
    // \0 to terminate.
    alt env {
      some (es) {
        let mut blk : [u8] = [];
        for (k,v) in es {
            let t = #fmt("%s=%s", k, v);
            let mut v : [u8] = ::unsafe::reinterpret_cast(t);
            blk += v;
            ::unsafe::leak(v);
        }
        blk += [0_u8];
        vec::as_buf(blk) {|p| cb(::unsafe::reinterpret_cast(p)) }
      }
      none {
        cb(ptr::null())
      }
    }
}

fn with_dirp<T>(d: option<str>,
                cb: fn(*u8) -> T) -> T unsafe {
    alt d {
      some(dir) { str::as_buf(dir, cb) }
      none { cb(ptr::null()) }
    }
}

#[doc ="
Spawns a process and waits for it to terminate

# Arguments

* prog - The path to an executable
* args - Vector of arguments to pass to the child process

# Return value

The process id
"]
fn run_program(prog: str, args: [str]) -> int {
    ret waitpid(spawn_process(prog, args, none, none,
                              0i32, 0i32, 0i32));
}

#[doc ="
Spawns a process and returns a program

The returned value is a boxed resource containing a <program> object that can
be used for sending and recieving data over the standard file descriptors.
The resource will ensure that file descriptors are closed properly.

# Arguments

* prog - The path to an executable
* args - Vector of arguments to pass to the child process

# Return value

A boxed resource of <program>
"]
fn start_program(prog: str, args: [str]) -> program {
    let pipe_input = os::pipe();
    let pipe_output = os::pipe();
    let pipe_err = os::pipe();
    let pid =
        spawn_process(prog, args, none, none,
                      pipe_input.in, pipe_output.out,
                      pipe_err.out);

    if pid == -1i32 { fail; }
    libc::close(pipe_input.in);
    libc::close(pipe_output.out);
    libc::close(pipe_err.out);

    type prog_repr = {pid: pid_t,
                      mutable in_fd: c_int,
                      out_file: *libc::FILE,
                      err_file: *libc::FILE,
                      mutable finished: bool};

    fn close_repr_input(r: prog_repr) {
        let invalid_fd = -1i32;
        if r.in_fd != invalid_fd {
            libc::close(r.in_fd);
            r.in_fd = invalid_fd;
        }
    }
    fn finish_repr(r: prog_repr) -> int {
        if r.finished { ret 0; }
        r.finished = true;
        close_repr_input(r);
        ret waitpid(r.pid);
    }
    fn destroy_repr(r: prog_repr) {
        finish_repr(r);
       libc::fclose(r.out_file);
       libc::fclose(r.err_file);
    }
    resource prog_res(r: prog_repr) { destroy_repr(r); }

    impl of program for prog_res {
        fn get_id() -> pid_t { ret self.pid; }
        fn input() -> io::writer { io::fd_writer(self.in_fd, false) }
        fn output() -> io::reader { io::FILE_reader(self.out_file, false) }
        fn err() -> io::reader { io::FILE_reader(self.err_file, false) }
        fn close_input() { close_repr_input(*self); }
        fn finish() -> int { finish_repr(*self) }
        fn destroy() { destroy_repr(*self); }
    }
    let repr = {pid: pid,
                mutable in_fd: pipe_input.out,
                out_file: os::fdopen(pipe_output.in),
                err_file: os::fdopen(pipe_err.in),
                mutable finished: false};
    ret prog_res(repr) as program;
}

fn read_all(rd: io::reader) -> str {
    let mut buf = "";
    while !rd.eof() {
        let bytes = rd.read_bytes(4096u);
        buf += str::from_bytes(bytes);
    }
    ret buf;
}

#[doc ="
Spawns a process, waits for it to exit, and returns the exit code, and
contents of stdout and stderr.

# Arguments

* prog - The path to an executable
* args - Vector of arguments to pass to the child process

# Return value

A record, {status: int, out: str, err: str} containing the exit code,
the contents of stdout and the contents of stderr.
"]
fn program_output(prog: str, args: [str]) ->
   {status: int, out: str, err: str} {
    let pr = start_program(prog, args);
    pr.close_input();
    let out = read_all(pr.output());
    let err = read_all(pr.err());
    ret {status: pr.finish(), out: out, err: err};
}

#[doc ="Waits for a process to exit and returns the exit code"]
fn waitpid(pid: pid_t) -> int {
    ret waitpid_os(pid);

    #[cfg(target_os = "win32")]
    fn waitpid_os(pid: pid_t) -> int {
        os::waitpid(pid) as int
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn waitpid_os(pid: pid_t) -> int {
        #[cfg(target_os = "linux")]
        fn WIFEXITED(status: i32) -> bool {
            (status & 0xffi32) == 0i32
        }

        #[cfg(target_os = "macos")]
        #[cfg(target_os = "freebsd")]
        fn WIFEXITED(status: i32) -> bool {
            (status & 0x7fi32) == 0i32
        }

        #[cfg(target_os = "linux")]
        fn WEXITSTATUS(status: i32) -> i32 {
            (status >> 8i32) & 0xffi32
        }

        #[cfg(target_os = "macos")]
        #[cfg(target_os = "freebsd")]
        fn WEXITSTATUS(status: i32) -> i32 {
            status >> 8i32
        }

        let status = os::waitpid(pid);
        ret if WIFEXITED(status) {
            WEXITSTATUS(status) as int
        } else {
            1
        };
    }
}

#[cfg(test)]
mod tests {

    import io::writer_util;

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
            run::spawn_process(
                "cat", [], none, none,
                pipe_in.in, pipe_out.out, pipe_err.out);
        os::close(pipe_in.in);
        os::close(pipe_out.out);
        os::close(pipe_err.out);

        if pid == -1i32 { fail; }
        let expected = "test";
        writeclose(pipe_in.out, expected);
        let actual = readclose(pipe_out.in);
        readclose(pipe_err.in);
        os::waitpid(pid);

        log(debug, expected);
        log(debug, actual);
        assert (expected == actual);

        fn writeclose(fd: c_int, s: str) {
            #error("writeclose %d, %s", fd as int, s);
            let writer = io::fd_writer(fd, false);
            writer.write_str(s);

            os::close(fd);
        }

        fn readclose(fd: c_int) -> str {
            // Copied from run::program_output
            let file = os::fdopen(fd);
            let reader = io::FILE_reader(file, false);
            let buf = "";
            while !reader.eof() {
                let bytes = reader.read_bytes(4096u);
                buf += str::from_bytes(bytes);
            }
            os::fclose(file);
            ret buf;
        }
    }

    #[test]
    fn waitpid() {
        let pid = run::spawn_process("false", [],
                                     none, none,
                                     0i32, 0i32, 0i32);
        let status = run::waitpid(pid);
        assert status == 1;
    }

}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
