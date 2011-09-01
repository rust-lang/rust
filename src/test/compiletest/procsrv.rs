// So when running tests in parallel there's a potential race on environment
// variables if we let each task spawn its own children - between the time the
// environment is set and the process is spawned another task could spawn its
// child process. Because of that we have to use a complicated scheme with a
// dedicated server for spawning processes.

import std::option;
import std::task;
import std::generic_os::setenv;
import std::generic_os::getenv;
import std::vec;
import std::os;
import std::run;
import std::io;
import std::istr;
import std::comm::chan;
import std::comm::port;
import std::comm::send;
import std::comm::recv;

export handle;
export mk;
export from_chan;
export run;
export close;
export reqchan;

type reqchan = chan<request>;

type handle = {task: option::t<(task::task, port<task::task_notification>)>,
    chan: reqchan};

tag request { exec([u8], [u8], [[u8]], chan<response>); stop; }

type response = {pid: int, infd: int, outfd: int, errfd: int};

fn mk() -> handle {
    let setupport = port();
    let task =
        task::spawn_joinable(bind fn (setupchan: chan<chan<request>>) {
            let reqport = port();
            let reqchan = chan(reqport);
            send(setupchan, reqchan);
            worker(reqport);
        }(chan(setupport)));
    ret {task: option::some(task), chan: recv(setupport)};
}

fn from_chan(ch: &reqchan) -> handle { {task: option::none, chan: ch} }

fn close(handle: &handle) {
    send(handle.chan, stop);
    task::join(option::get(handle.task));
}

fn run(handle: &handle, lib_path: &istr, prog: &istr, args: &[istr],
       input: &option::t<istr>) -> {status: int, out: istr, err: istr} {
    let p = port();
    let ch = chan(p);
    send(handle.chan,
         exec(istr::bytes(lib_path), istr::bytes(prog), clone_vecstr(args),
              ch));
    let resp = recv(p);

    writeclose(resp.infd, input);
    let output = readclose(resp.outfd);
    let errput = readclose(resp.errfd);
    let status = os::waitpid(resp.pid);
    ret {status: status, out: output, err: errput};
}

fn writeclose(fd: int, s: &option::t<istr>) {
    if option::is_some(s) {
        let writer = io::new_writer(io::fd_buf_writer(fd, option::none));
        writer.write_str(option::get(s));
    }

    os::libc::close(fd);
}

fn readclose(fd: int) -> istr {
    // Copied from run::program_output
    let file = os::fd_FILE(fd);
    let reader = io::new_reader(io::FILE_buf_reader(file, option::none));
    let buf = ~"";
    while !reader.eof() {
        let bytes = reader.read_bytes(4096u);
        buf += istr::unsafe_from_bytes(bytes);
    }
    os::libc::fclose(file);
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
                    {lib_path: istr::unsafe_from_bytes(lib_path),
                     prog: istr::unsafe_from_bytes(prog),
                     args: clone_vecu8str(args),
                     respchan: respchan}
                  }
                  stop. { ret }
                }
            };

        // This is copied from run::start_program
        let pipe_in = os::pipe();
        let pipe_out = os::pipe();
        let pipe_err = os::pipe();
        let spawnproc =
            bind run::spawn_process(execparms.prog,
                                    execparms.args,
                                    pipe_in.in, pipe_out.out, pipe_err.out);
        let pid = with_lib_path(execparms.lib_path, spawnproc);

        os::libc::close(pipe_in.in);
        os::libc::close(pipe_out.out);
        os::libc::close(pipe_err.out);
        if pid == -1 {
            os::libc::close(pipe_in.out);
            os::libc::close(pipe_out.in);
            os::libc::close(pipe_err.in);
            fail;
        }

        send(execparms.respchan,
             {pid: pid,
              infd: pipe_in.out,
              outfd: pipe_out.in,
              errfd: pipe_err.in});
    }
}

fn with_lib_path<@T>(path: &istr, f: fn() -> T) -> T {
    let maybe_oldpath = getenv(util::lib_path_env_var());
    append_lib_path(path);
    let res = f();
    if option::is_some(maybe_oldpath) {
        export_lib_path(option::get(maybe_oldpath));
    } else {
        // FIXME: This should really be unset but we don't have that yet
        export_lib_path(~"");
    }
    ret res;
}

fn append_lib_path(path: &istr) {
    export_lib_path(util::make_new_path(path));
}

fn export_lib_path(path: &istr) {
    setenv(util::lib_path_env_var(), path);
}

fn clone_vecstr(v: &[istr]) -> [[u8]] {
    let r = [];
    for t: istr in vec::slice(v, 0u, vec::len(v)) { r += [istr::bytes(t)]; }
    ret r;
}

fn clone_vecu8str(v: &[[u8]]) -> [istr] {
    let r = [];
    for t in vec::slice(v, 0u, vec::len(v)) {
        r += [istr::unsafe_from_bytes(t)];
    }
    ret r;
}
