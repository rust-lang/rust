import Str.sbuf;
import Vec.vbuf;

native "rust" mod rustrt {
    fn rust_run_program(vbuf argv, int in_fd, int out_fd, int err_fd) -> int;
}

fn argvec(str prog, vec[str] args) -> vec[sbuf] {
    auto argptrs = vec(Str.buf(prog));
    for (str arg in args) {
        Vec.push[sbuf](argptrs, Str.buf(arg));
    }
    Vec.push[sbuf](argptrs, 0 as sbuf);
    ret argptrs;
}

fn run_program(str prog, vec[str] args) -> int {
    auto pid = rustrt.rust_run_program(Vec.buf[sbuf](argvec(prog, args)),
                                       0, 0, 0);
    ret OS.waitpid(pid);
}

type program =
    state obj {
        fn get_id() -> int;
        fn input() -> IO.writer;
        fn output() -> IO.reader;
        fn close_input();
        fn finish() -> int;
    };

fn start_program(str prog, vec[str] args) -> @program {
    auto pipe_input = OS.pipe();
    auto pipe_output = OS.pipe();
    auto pid = rustrt.rust_run_program
        (Vec.buf[sbuf](argvec(prog, args)),
         pipe_input._0, pipe_output._1, 0);
    if (pid == -1) {fail;}
    OS.libc.close(pipe_input._0);
    OS.libc.close(pipe_output._1);

    state obj new_program(int pid,
                          int in_fd,
                          OS.libc.FILE out_file,
                          mutable bool finished) {
        fn get_id() -> int {ret pid;}
        fn input() -> IO.writer {
            ret IO.new_writer(IO.fd_buf_writer(in_fd, false));
        }
        fn output() -> IO.reader {
            ret IO.new_reader(IO.FILE_buf_reader(out_file, false));
        }
        fn close_input() {
            OS.libc.close(in_fd);
        }
        fn finish() -> int {
            if (finished) {ret 0;}
            finished = true;
            OS.libc.close(in_fd);
            ret OS.waitpid(pid);
        }
        drop {
            if (!finished) {
                OS.libc.close(in_fd);
                OS.waitpid(pid);
            }
            OS.libc.fclose(out_file);
        }
    }
    ret @new_program(pid, pipe_input._1,
                     OS.fd_FILE(pipe_output._0),
                     false);
}

fn program_output(str prog, vec[str] args)
    -> rec(int status, str out) {
    auto pr = start_program(prog, args);
    pr.close_input();
    auto out = pr.output();
    auto buf = "";
    while (!out.eof()) {
        auto bytes = out.read_bytes(4096u);
        buf += Str.unsafe_from_bytes(bytes);
    }
    ret rec(status=pr.finish(), out=buf);
}


// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
