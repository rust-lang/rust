
import str::sbuf;

export program;
export run_program;
export start_program;
export program_output;
export spawn_process;

native "rust" mod rustrt {
    fn rust_run_program(argv: *sbuf, in_fd: int, out_fd: int, err_fd: int) ->
       int;
}

fn arg_vec(prog: str, args: &[str]) -> [sbuf] {
    let argptrs = [str::buf(prog)];
    for arg: str in args { argptrs += [str::buf(arg)]; }
    argptrs += [0 as sbuf];
    ret argptrs;
}

fn spawn_process(prog: &istr, args: &[istr], in_fd: int, out_fd: int,
                 err_fd: int) -> int {
    let prog = istr::to_estr(prog);
    let args = istr::to_estrs(args);
    // Note: we have to hold on to this vector reference while we hold a
    // pointer to its buffer
    let argv = arg_vec(prog, args);
    let pid =
        rustrt::rust_run_program(vec::to_ptr(argv), in_fd, out_fd, err_fd);
    ret pid;
}

fn run_program(prog: &istr, args: &[istr]) -> int {
    ret os::waitpid(spawn_process(prog, args, 0, 0, 0));
}

type program =
    obj {
        fn get_id() -> int;
        fn input() -> io::writer;
        fn output() -> io::reader;
        fn err() -> io::reader;
        fn close_input();
        fn finish() -> int;
        fn destroy();
    };

resource program_res(p: program) { p.destroy(); }

fn start_program(prog: &istr, args: &[istr]) -> @program_res {
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
            ret os::waitpid(pid);
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

fn read_all(rd: &io::reader) -> str {
    let buf = "";
    while !rd.eof() {
        let bytes = rd.read_bytes(4096u);
        buf += str::unsafe_from_bytes(bytes);
    }
    ret buf;
}

fn program_output(prog: &istr, args: &[istr]) ->
   {status: int, out: str, err: str} {
    let pr = start_program(prog, args);
    pr.close_input();
    ret {status: pr.finish(),
         out: read_all(pr.output()),
         err: read_all(pr.err())};
}
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
