use run::spawn_process;
use io::{ReaderUtil, WriterUtil};
use libc::{c_int, pid_t};

export run;

#[cfg(target_os = "win32")]
fn target_env(lib_path: ~str, prog: ~str) -> ~[(~str,~str)] {

    let mut env = os::env();

    // Make sure we include the aux directory in the path
    assert prog.ends_with(~".exe");
    let aux_path = prog.slice(0u, prog.len() - 4u) + ~".libaux";

    env = do vec::map(env) |pair| {
        let (k,v) = pair;
        if k == ~"PATH" { (~"PATH", v + ~";" + lib_path + ~";" + aux_path) }
        else { (k,v) }
    };
    if str::ends_with(prog, ~"rustc.exe") {
        vec::push(env, (~"RUST_THREADS", ~"1"));
    }
    return env;
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
fn target_env(_lib_path: ~str, _prog: ~str) -> ~[(~str,~str)] {
    ~[]
}


// FIXME (#2659): This code is duplicated in core::run::program_output
fn run(lib_path: ~str,
       prog: ~str,
       args: ~[~str],
       env: ~[(~str, ~str)],
       input: Option<~str>) -> {status: int, out: ~str, err: ~str} {

    let pipe_in = os::pipe();
    let pipe_out = os::pipe();
    let pipe_err = os::pipe();
    let pid = spawn_process(prog, args,
                            &Some(env + target_env(lib_path, prog)),
                            &None, pipe_in.in, pipe_out.out, pipe_err.out);

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
    let p = pipes::PortSet();
    let ch = p.chan();
    do task::spawn_sched(task::SingleThreaded) {
        let errput = readclose(pipe_err.in);
        ch.send((2, errput));
    }
    let ch = p.chan();
    do task::spawn_sched(task::SingleThreaded) {
        let output = readclose(pipe_out.in);
        ch.send((1, output));
    }
    let status = run::waitpid(pid);
    let mut errs = ~"";
    let mut outs = ~"";
    let mut count = 2;
    while count > 0 {
        match p.recv() {
          (1, s) => {
            outs = s;
          }
          (2, s) => {
            errs = s;
          }
          _ => { fail }
        };
        count -= 1;
    };
    return {status: status, out: outs, err: errs};
}

fn writeclose(fd: c_int, s: Option<~str>) {
    if s.is_some() {
        let writer = io::fd_writer(fd, false);
        writer.write_str(s.get());
    }

    os::close(fd);
}

fn readclose(fd: c_int) -> ~str {
    // Copied from run::program_output
    let file = os::fdopen(fd);
    let reader = io::FILE_reader(file, false);
    let mut buf = ~"";
    while !reader.eof() {
        let bytes = reader.read_bytes(4096u);
        str::push_str(&mut buf, str::from_bytes(bytes));
    }
    os::fclose(file);
    return buf;
}
