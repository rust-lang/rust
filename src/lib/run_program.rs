import _str.sbuf;
import _vec.vbuf;

native "rust" mod rustrt {
    fn rust_run_program(vbuf argv, int in_fd, int out_fd, int err_fd) -> int;
}

fn argvec(str prog, vec[str] args) -> vec[sbuf] {
    auto argptrs = vec(_str.buf(prog));
    for (str arg in args) {
        _vec.push[sbuf](argptrs, _str.buf(arg));
    }
    _vec.push[sbuf](argptrs, 0 as sbuf);
    ret argptrs;
}

impure fn run_program(str prog, vec[str] args) -> int {
    auto pid = rustrt.rust_run_program(_vec.buf[sbuf](argvec(prog, args)),
                                       0, 0, 0);
    ret os.waitpid(pid);
}

type program =
    state obj {
        fn get_id() -> int;
        fn input() -> io.writer;
        fn output() -> io.reader;
        impure fn close_input();
        impure fn finish() -> int;
    };

impure fn start_program(str prog, vec[str] args) -> @program {
    auto pipe_input = os.pipe();
    auto pipe_output = os.pipe();
    auto pid = rustrt.rust_run_program
        (_vec.buf[sbuf](argvec(prog, args)),
         pipe_input._0, pipe_output._1, 0);
    if (pid == -1) {fail;}
    os.libc.close(pipe_input._0);
    os.libc.close(pipe_output._1);

    state obj new_program(int pid,
                          int in_fd,
                          os.libc.FILE out_file,
                          mutable bool finished) {
        fn get_id() -> int {ret pid;}
        fn input() -> io.writer {
            ret io.new_writer(io.fd_buf_writer(in_fd, false));
        }
        fn output() -> io.reader {
            ret io.FILE_reader(out_file, false);
        }
        impure fn close_input() {
            os.libc.close(in_fd);
        }
        impure fn finish() -> int {
            if (finished) {ret 0;}
            finished = true;
            os.libc.close(in_fd);
            ret os.waitpid(pid);
        }
        drop {
            if (!finished) {
                os.libc.close(in_fd);
                os.waitpid(pid);
            }
            os.libc.fclose(out_file);
        }
    }
    ret @new_program(pid, pipe_input._1,
                     os.fd_FILE(pipe_output._0),
                     false);
}

impure fn program_output(str prog, vec[str] args)
    -> rec(int status, str out) {
    auto pr = start_program(prog, args);
    pr.close_input();
    auto out = pr.output();
    auto buf = "";
    while (!out.eof()) {
        auto bytes = out.read_bytes(4096u);
        buf += _str.unsafe_from_bytes(bytes);
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
