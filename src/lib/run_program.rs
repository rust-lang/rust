/*
Module: run

Process spawning
*/
import str::sbuf;

export program;
export run_program;
export start_program;
export program_output;
export spawn_process;
export waitpid;

native "cdecl" mod rustrt {
    fn rust_run_program(argv: *sbuf, in_fd: int, out_fd: int, err_fd: int) ->
       int;
}

/* Section: Types */

/*
Resource: program_res

A resource that manages the destruction of a <program> object

program_res ensures that the destroy method is called on a
program object in order to close open file descriptors.
*/
resource program_res(p: program) { p.destroy(); }

/*
Obj: program

An object representing a child process
*/
type program = obj {
    /*
    Method: get_id

    Returns the process id of the program
    */
    fn get_id() -> int;

    /*
    Method: input

    Returns an io::writer that can be used to write to stdin
    */
    fn input() -> io::writer;

    /*
    Method: output

    Returns an io::reader that can be used to read from stdout
    */
    fn output() -> io::reader;

    /*
    Method: err

    Returns an io::reader that can be used to read from stderr
    */
    fn err() -> io::reader;

    /*
    Method: close_input

    Closes the handle to the child processes standard input
    */
    fn close_input();

    /*
    Method: finish

    Waits for the child process to terminate. Closes the handle
    to stdin if necessary.
    */
    fn finish() -> int;

    /*
    Method: destroy

    Closes open handles
    */
    fn destroy();
};


/* Section: Operations */

fn arg_vec(prog: str, args: [@str]) -> [sbuf] {
    let argptrs = str::as_buf(prog, {|buf| [buf] });
    for arg in args { argptrs += str::as_buf(*arg, {|buf| [buf] }); }
    argptrs += [ptr::null()];
    ret argptrs;
}

/*
Function: spawn_process

Run a program, providing stdin, stdout and stderr handles

Parameters:

prog - The path to an executable
args - Vector of arguments to pass to the child process
in_fd - A file descriptor for the child to use as std input
out_fd - A file descriptor for the child to use as std output
err_fd - A file descriptor for the child to use as std error

Returns:

The process id of the spawned process
*/
fn spawn_process(prog: str, args: [str], in_fd: int, out_fd: int, err_fd: int)
   -> int unsafe {
    // Note: we have to hold on to these vector references while we hold a
    // pointer to their buffers
    let prog = prog;
    let args = vec::map({|arg| @arg }, args);
    let argv = arg_vec(prog, args);
    let pid =
        rustrt::rust_run_program(vec::unsafe::to_ptr(argv), in_fd, out_fd,
                                 err_fd);
    ret pid;
}

/*
Function: run_program

Spawns a process and waits for it to terminate

Parameters:

prog - The path to an executable
args - Vector of arguments to pass to the child process

Returns:

The process id
*/
fn run_program(prog: str, args: [str]) -> int {
    ret waitpid(spawn_process(prog, args, 0, 0, 0));
}

/*
Function: start_program

Spawns a process and returns a boxed <program_res>

The returned value is a boxed resource containing a <program> object that can
be used for sending and recieving data over the standard file descriptors.
The resource will ensure that file descriptors are closed properly.

Parameters:

prog - The path to an executable
args - Vector of arguments to pass to the child process

Returns:

A boxed resource of <program>
*/
fn start_program(prog: str, args: [str]) -> @program_res {
    let pipe_input = os::pipe();
    let pipe_output = os::pipe();
    let pipe_err = os::pipe();
    let pid =
        spawn_process(prog, args, pipe_input.in, pipe_output.out,
                      pipe_err.out);

    if pid == -1 { fail; }
    os::libc::close(pipe_input.in);
    os::libc::close(pipe_output.out);
    os::libc::close(pipe_err.out);
    obj new_program(pid: int,
                    mutable in_fd: int,
                    out_file: os::libc::FILE,
                    err_file: os::libc::FILE,
                    mutable finished: bool) {
        fn get_id() -> int { ret pid; }
        fn input() -> io::writer {
            ret io::new_writer(io::fd_buf_writer(in_fd, option::none));
        }
        fn output() -> io::reader {
            ret io::new_reader(io::FILE_buf_reader(out_file, option::none));
        }
        fn err() -> io::reader {
            ret io::new_reader(io::FILE_buf_reader(err_file, option::none));
        }
        fn close_input() {
            let invalid_fd = -1;
            if in_fd != invalid_fd {
                os::libc::close(in_fd);
                in_fd = invalid_fd;
            }
        }
        fn finish() -> int {
            if finished { ret 0; }
            finished = true;
            self.close_input();
            ret waitpid(pid);
        }
        fn destroy() {
            self.finish();
            os::libc::fclose(out_file);
            os::libc::fclose(err_file);
        }
    }
    ret @program_res(new_program(pid, pipe_input.out,
                                 os::fd_FILE(pipe_output.in),
                                 os::fd_FILE(pipe_err.in), false));
}

fn read_all(rd: io::reader) -> str {
    let buf = "";
    while !rd.eof() {
        let bytes = rd.read_bytes(4096u);
        buf += str::unsafe_from_bytes(bytes);
    }
    ret buf;
}

/*
Function: program_output

Spawns a process, waits for it to exit, and returns the exit code, and
contents of stdout and stderr.

Parameters:

prog - The path to an executable
args - Vector of arguments to pass to the child process

Returns:

A record, {status: int, out: str, err: str} containing the exit code,
the contents of stdout and the contents of stderr.
*/
fn program_output(prog: str, args: [str]) ->
   {status: int, out: str, err: str} {
    let pr = start_program(prog, args);
    pr.close_input();
    let out = read_all(pr.output());
    let err = read_all(pr.err());
    ret {status: pr.finish(), out: out, err: err};
}

/*
Function: waitpid

Waits for a process to exit and returns the exit code
*/
fn waitpid(pid: int) -> int {
    ret waitpid_os(pid);

    #[cfg(target_os = "win32")]
    fn waitpid_os(pid: int) -> int {
        os::waitpid(pid)
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    fn waitpid_os(pid: int) -> int {
        #[cfg(target_os = "linux")]
        fn WIFEXITED(status: int) -> bool {
            (status & 0xff) == 0
        }

        #[cfg(target_os = "macos")]
        fn WIFEXITED(status: int) -> bool {
            (status & 0x7f) == 0
        }

        #[cfg(target_os = "linux")]
        fn WEXITSTATUS(status: int) -> int {
            (status >> 8) & 0xff
        }

        #[cfg(target_os = "macos")]
        fn WEXITSTATUS(status: int) -> int {
            status >> 8
        }

        let status = os::waitpid(pid);
        ret if WIFEXITED(status) {
            WEXITSTATUS(status)
        } else {
            1
        };
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
