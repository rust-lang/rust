
import str::sbuf;
import vec::vbuf;

native "rust" mod rustrt {
    fn rust_run_program(vbuf argv, int in_fd, int out_fd, int err_fd) -> int;
}

fn arg_vec(str prog, vec[str] args) -> vec[sbuf] {
    auto argptrs = [str::buf(prog)];
    for (str arg in args) { vec::push[sbuf](argptrs, str::buf(arg)); }
    vec::push[sbuf](argptrs, 0 as sbuf);
    ret argptrs;
}

fn run_program(str prog, vec[str] args) -> int {
    auto pid =
        rustrt::rust_run_program(vec::buf[sbuf](arg_vec(prog, args)), 0, 0,
                                 0);
    ret os::waitpid(pid);
}

type program =
    obj {
        fn get_id() -> int ;
        fn input() -> io::writer ;
        fn output() -> io::reader ;
        fn close_input() ;
        fn finish() -> int ;
    };

fn start_program(str prog, vec[str] args) -> @program {
    auto pipe_input = os::pipe();
    auto pipe_output = os::pipe();
    auto pid =
        rustrt::rust_run_program(vec::buf[sbuf](arg_vec(prog, args)),
                                 pipe_input._0, pipe_output._1, 0);
    if (pid == -1) { fail; }
    os::libc::close(pipe_input._0);
    os::libc::close(pipe_output._1);
    obj new_program(int pid,
                    int in_fd,
                    os::libc::FILE out_file,
                    mutable bool finished) {
        fn get_id() -> int { ret pid; }
        fn input() -> io::writer {
            ret io::new_writer(io::fd_buf_writer(in_fd, false));
        }
        fn output() -> io::reader {
            ret io::new_reader(io::FILE_buf_reader(out_file, false));
        }
        fn close_input() { os::libc::close(in_fd); }
        fn finish() -> int {
            if (finished) { ret 0; }
            finished = true;
            os::libc::close(in_fd);
            ret os::waitpid(pid);
        }drop {
             if (!finished) { os::libc::close(in_fd); os::waitpid(pid); }
             os::libc::fclose(out_file);
         }
    }
    ret @new_program(pid, pipe_input._1, os::fd_FILE(pipe_output._0), false);
}

fn program_output(str prog, vec[str] args) -> rec(int status, str out) {
    auto pr = start_program(prog, args);
    pr.close_input();
    auto out = pr.output();
    auto buf = "";
    while (!out.eof()) {
        auto bytes = out.read_bytes(4096u);
        buf += str::unsafe_from_bytes(bytes);
    }
    ret rec(status=pr.finish(), out=buf);
}
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
