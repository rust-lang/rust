// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use core::libc::c_int;
use core::run::spawn_process;
use core::run;

#[cfg(target_os = "win32")]
fn target_env(lib_path: &str, prog: &str) -> ~[(~str,~str)] {

    let mut env = os::env();

    // Make sure we include the aux directory in the path
    assert!(prog.ends_with(~".exe"));
    let aux_path = prog.slice(0u, prog.len() - 4u).to_owned() + ~".libaux";

    env = do vec::map(env) |pair| {
        let (k,v) = *pair;
        if k == ~"PATH" { (~"PATH", v + ~";" + lib_path + ~";" + aux_path) }
        else { (k,v) }
    };
    if str::ends_with(prog, "rustc.exe") {
        env.push((~"RUST_THREADS", ~"1"));
    }
    return env;
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
fn target_env(_lib_path: &str, _prog: &str) -> ~[(~str,~str)] {
    ~[]
}

pub struct Result {status: int, out: ~str, err: ~str}

// FIXME (#2659): This code is duplicated in core::run::program_output
pub fn run(lib_path: &str,
           prog: &str,
           args: &[~str],
           env: ~[(~str, ~str)],
           input: Option<~str>) -> Result {
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
        fail!();
    }


    writeclose(pipe_in.out, input);
    let p = comm::PortSet::new();
    let ch = p.chan();
    do task::spawn_sched(task::SingleThreaded) || {
        let errput = readclose(pipe_err.in);
        ch.send((2, errput));
    }
    let ch = p.chan();
    do task::spawn_sched(task::SingleThreaded) || {
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
          _ => { fail!() }
        };
        count -= 1;
    };
    return Result {status: status, out: outs, err: errs};
}

fn writeclose(fd: c_int, s: Option<~str>) {
    if s.is_some() {
        let writer = io::fd_writer(fd, false);
        writer.write_str(s.get());
    }

    os::close(fd);
}

fn readclose(fd: c_int) -> ~str {
    unsafe {
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
}
