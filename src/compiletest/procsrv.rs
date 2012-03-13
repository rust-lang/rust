import run::spawn_process;
import io::writer_util;
import libc::{c_int, pid_t};

export run;

#[cfg(target_os = "win32")]
fn target_env(lib_path: str, prog: str) -> option<[(str,str)]> {

    let env = os::env();

    env = vec::map(env) {|pair|
        let (k,v) = pair;
        if k == "PATH" { ("PATH", v + ";" + lib_path) }
        else { (k,v) }
    };
    if str::ends_with(prog, "rustc.exe") {
        env += [("RUST_THREADS", "1")]
    }
    ret some(env);
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
fn target_env(_lib_path: str, _prog: str) -> option<[(str,str)]> {
    none
}


fn run(lib_path: str, prog: str, args: [str],
       input: option<str>) -> {status: int, out: str, err: str} {

    let pipe_in = os::pipe();
    let pipe_out = os::pipe();
    let pipe_err = os::pipe();
    let pid = spawn_process(prog, args, target_env(lib_path, prog), none,
                            pipe_in.in, pipe_out.out, pipe_err.out);

    os::close(pipe_in.in);
    os::close(pipe_out.out);
    os::close(pipe_err.out);
    if pid == -1i32 {
        os::close(pipe_in.out);
        os::close(pipe_out.in);
        os::close(pipe_err.in);
        fail;
    }


    writeclose(pipe_in.out, input);
    let p = comm::port();
    let ch = comm::chan(p);
    task::spawn_sched(task::single_threaded) {||
        let errput = readclose(pipe_err.in);
        comm::send(ch, (2, errput));
    };
    task::spawn_sched(task::single_threaded) {||
        let output = readclose(pipe_out.in);
        comm::send(ch, (1, output));
    };
    let status = run::waitpid(pid);
    let errs = "";
    let outs = "";
    let count = 2;
    while count > 0 {
        let stream = comm::recv(p);
        alt check stream {
            (1, s) {
                outs = s;
            }
            (2, s) {
                errs = s;
            }
        };
        count -= 1;
    };
    ret {status: status, out: outs, err: errs};
}

fn writeclose(fd: c_int, s: option<str>) {
    if option::is_some(s) {
        let writer = io::fd_writer(fd, false);
        writer.write_str(option::get(s));
    }

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
