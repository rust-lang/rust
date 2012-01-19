// So when running tests in parallel there's a potential race on environment
// variables if we let each task spawn its own children - between the time the
// environment is set and the process is spawned another task could spawn its
// child process. Because of that we have to use a complicated scheme with a
// dedicated server for spawning processes.

import std::generic_os::setenv;
import std::generic_os::getenv;
import std::os;
import std::run;
import std::io;
import io::writer_util;
import comm::chan;
import comm::port;
import comm::send;
import comm::recv;
import ctypes::{pid_t, fd_t};

export handle;
export mk;
export from_chan;
export run;
export close;
export reqchan;

type reqchan = chan<request>;

type handle =
    {task: option::t<(task::task, port<task::task_notification>)>,
     chan: reqchan};

tag request { exec([u8], [u8], [[u8]], chan<response>); stop; }

type response = {pid: pid_t, infd: fd_t,
                 outfd: fd_t, errfd: fd_t};

fn mk() -> handle {
    let setupport = port();
    let setupchan = chan(setupport);
    let task = task::spawn_joinable {||
        let reqport = port();
        let reqchan = chan(reqport);
        send(setupchan, reqchan);
        worker(reqport);
    };
    ret {task: option::some(task), chan: recv(setupport)};
}

fn from_chan(ch: reqchan) -> handle { {task: option::none, chan: ch} }

fn close(handle: handle) {
    send(handle.chan, stop);
    task::join(option::get(handle.task));
}

fn run(handle: handle, lib_path: str, prog: str, args: [str],
       input: option::t<str>) -> {status: int, out: str, err: str} {
    let p = port();
    let ch = chan(p);
    send(handle.chan,
         exec(str::bytes(lib_path), str::bytes(prog), clone_vecstr(args),
              ch));
    let resp = recv(p);

    writeclose(resp.infd, input);
    let output = readclose(resp.outfd);
    let errput = readclose(resp.errfd);
    let status = run::waitpid(resp.pid);
    ret {status: status, out: output, err: errput};
}

fn writeclose(fd: fd_t, s: option::t<str>) {
    if option::is_some(s) {
        let writer = io::fd_writer(fd, false);
        writer.write_str(option::get(s));
    }

    os::close(fd);
}

fn readclose(fd: fd_t) -> str {
    // Copied from run::program_output
    let file = os::fd_FILE(fd);
    let reader = io::FILE_reader(file, false);
    let buf = "";
    while !reader.eof() {
        let bytes = reader.read_bytes(4096u);
        buf += str::unsafe_from_bytes(bytes);
    }
    os::fclose(file);
    ret buf;
}

fn worker(p: port<request>) {

    // FIXME (787): If we declare this inside of the while loop and then
    // break out of it before it's ever initialized (i.e. we don't run
    // any tests), then the cleanups will puke.
    let execparms;

    while true {
        // FIXME: Sending strings across channels seems to still
        // leave them refed on the sender's end, which causes problems if
        // the receiver's poniters outlive the sender's. Here we clone
        // everything and let the originals go out of scope before sending
        // a response.
        execparms =
            {

                // FIXME (785): The 'discriminant' of an alt expression has
                // the same scope as the alt expression itself, so we have to
                // put the entire alt in another block to make sure the exec
                // message goes out of scope. Seems like the scoping rules for
                // the alt discriminant are wrong.
                alt recv(p) {
                  exec(lib_path, prog, args, respchan) {
                    {lib_path: str::unsafe_from_bytes(lib_path),
                     prog: str::unsafe_from_bytes(prog),
                     args: clone_vecu8str(args),
                     respchan: respchan}
                  }
                  stop { ret }
                }
            };

        // This is copied from run::start_program
        let pipe_in = os::pipe();
        let pipe_out = os::pipe();
        let pipe_err = os::pipe();
        let spawnproc =
            bind run::spawn_process(execparms.prog, execparms.args,
                                    pipe_in.in, pipe_out.out, pipe_err.out);
        let pid = maybe_with_lib_path(execparms.lib_path, spawnproc);

        os::close(pipe_in.in);
        os::close(pipe_out.out);
        os::close(pipe_err.out);
        if pid == -1i32 {
            os::close(pipe_in.out);
            os::close(pipe_out.in);
            os::close(pipe_err.in);
            fail;
        }

        send(execparms.respchan,
             {pid: pid,
              infd: pipe_in.out,
              outfd: pipe_out.in,
              errfd: pipe_err.in});
    }
}

// Only windows needs to set the library path
#[cfg(target_os = "win32")]
fn maybe_with_lib_path<T>(path: str, f: fn@() -> T) -> T {
    with_lib_path(path, f)
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
fn maybe_with_lib_path<T>(_path: str, f: fn@() -> T) -> T {
    f()
}

fn with_lib_path<T>(path: str, f: fn@() -> T) -> T {
    let maybe_oldpath = getenv(util::lib_path_env_var());
    append_lib_path(path);
    let res = f();
    if option::is_some(maybe_oldpath) {
        export_lib_path(option::get(maybe_oldpath));
    } else {
        // FIXME: This should really be unset but we don't have that yet
        export_lib_path("");
    }
    ret res;
}

fn append_lib_path(path: str) { export_lib_path(util::make_new_path(path)); }

fn export_lib_path(path: str) { setenv(util::lib_path_env_var(), path); }

fn clone_vecstr(v: [str]) -> [[u8]] {
    let r = [];
    for t: str in vec::slice(v, 0u, vec::len(v)) { r += [str::bytes(t)]; }
    ret r;
}

fn clone_vecu8str(v: [[u8]]) -> [str] {
    let r = [];
    for t in vec::slice(v, 0u, vec::len(v)) {
        r += [str::unsafe_from_bytes(t)];
    }
    ret r;
}
