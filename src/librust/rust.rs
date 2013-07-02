// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// rust - central access to other rust tools
// FIXME #2238 Make commands run and test emit proper file endings on windows
// FIXME #2238 Make run only accept source that emits an executable

#[link(name = "rust",
       vers = "0.7",
       uuid = "4a24da33-5cc8-4037-9352-2cbe9bd9d27c",
       url = "https://github.com/mozilla/rust/tree/master/src/rust")];

#[license = "MIT/ASL2"];
#[crate_type = "lib"];

extern mod rustpkg;
extern mod rustdoc;
extern mod rusti;
extern mod rustc;

use std::io;
use std::os;
use std::run;
use std::libc::exit;

enum ValidUsage {
    Valid(int), Invalid
}

impl ValidUsage {
    fn is_valid(&self) -> bool {
        match *self {
            Valid(_)   => true,
            Invalid    => false
        }
    }
}

enum Action<'self> {
    Call(&'self fn:Copy(args: &[~str]) -> ValidUsage),
    CallMain(&'static str, &'self fn:Copy()),
}

enum UsageSource<'self> {
    UsgStr(&'self str),
    UsgCall(&'self fn:Copy()),
}

struct Command<'self> {
    cmd: &'self str,
    action: Action<'self>,
    usage_line: &'self str,
    usage_full: UsageSource<'self>,
}

static commands: &'static [Command<'static>] = &[
    Command{
        cmd: "build",
        action: CallMain("rustc", rustc::main),
        usage_line: "compile rust source files",
        usage_full: UsgCall(rustc_help),
    },
    Command{
        cmd: "run",
        action: Call(cmd_run),
        usage_line: "build an executable, and run it",
        usage_full: UsgStr(
            "The run command is an shortcut for the command line \n\
             \"rustc <filename> -o <filestem>~ && ./<filestem>~ [<arguments>...]\".\
            \n\nUsage:\trust run <filename> [<arguments>...]"
        )
    },
    Command{
        cmd: "test",
        action: Call(cmd_test),
        usage_line: "build a test executable, and run it",
        usage_full: UsgStr(
            "The test command is an shortcut for the command line \n\
            \"rustc --test <filename> -o <filestem>test~ && \
            ./<filestem>test~\"\n\nUsage:\trust test <filename>"
        )
    },
    Command{
        cmd: "doc",
        action: CallMain("rustdoc", rustdoc::main),
        usage_line: "generate documentation from doc comments",
        usage_full: UsgCall(rustdoc::config::usage),
    },
    Command{
        cmd: "pkg",
        action: CallMain("rustpkg", rustpkg::main),
        usage_line: "download, build, install rust packages",
        usage_full: UsgCall(rustpkg::usage::general),
    },
    Command{
        cmd: "sketch",
        action: CallMain("rusti", rusti::main),
        usage_line: "run a rust interpreter",
        usage_full: UsgStr("\nUsage:\trusti"),
    },
    Command{
        cmd: "help",
        action: Call(cmd_help),
        usage_line: "show detailed usage of a command",
        usage_full: UsgStr(
            "The help command displays the usage text of another command.\n\
            The text is either build in, or provided by the corresponding \
            program.\n\nUsage:\trust help <command>"
        )
    }
];

fn rustc_help() {
    rustc::usage(copy os::args()[0])
}

fn find_cmd(command_string: &str) -> Option<Command> {
    do commands.iter().find_ |command| {
        command.cmd == command_string
    }.map_consume(|x| copy *x)
}

fn cmd_help(args: &[~str]) -> ValidUsage {
    fn print_usage(command_string: ~str) -> ValidUsage {
        match find_cmd(command_string) {
            Some(command) => {
                match command.action {
                    CallMain(prog, _) => io::println(fmt!(
                        "The %s command is an alias for the %s program.",
                        command.cmd, prog)),
                    _       => ()
                }
                match command.usage_full {
                    UsgStr(msg) => io::println(fmt!("%s\n", msg)),
                    UsgCall(f)  => f(),
                }
                Valid(0)
            },
            None => Invalid
        }
    }

    match args {
        [ref command_string] => print_usage(copy *command_string),
        _                    => Invalid
    }
}

fn cmd_test(args: &[~str]) -> ValidUsage {
    match args {
        [ref filename] => {
            let test_exec = Path(*filename).filestem().unwrap() + "test~";
            invoke("rustc", &[~"--test", filename.to_owned(),
                              ~"-o", test_exec.to_owned()], rustc::main);
            let exit_code = run::process_status(~"./" + test_exec, []);
            Valid(exit_code)
        }
        _ => Invalid
    }
}

fn cmd_run(args: &[~str]) -> ValidUsage {
    match args {
        [ref filename, ..prog_args] => {
            let exec = Path(*filename).filestem().unwrap() + "~";
            invoke("rustc", &[filename.to_owned(), ~"-o", exec.to_owned()],
                   rustc::main);
            let exit_code = run::process_status(~"./"+exec, prog_args);
            Valid(exit_code)
        }
        _ => Invalid
    }
}

fn invoke(prog: &str, args: &[~str], f: &fn()) {
    let mut osargs = ~[prog.to_owned()];
    osargs.push_all_move(args.to_owned());
    os::set_args(osargs);
    f();
}

fn do_command(command: &Command, args: &[~str]) -> ValidUsage {
    match command.action {
        Call(f) => f(args),
        CallMain(prog, f) => {
            invoke(prog, args, f);
            Valid(0)
        }
    }
}

fn usage() {
    static indent: uint = 8;

    io::print(
        "The rust tool is a convenience for managing rust source code.\n\
        It acts as a shortcut for programs of the rust tool chain.\n\
        \n\
        Usage:\trust <command> [arguments]\n\
        \n\
        The commands are:\n\
        \n"
    );

    for commands.iter().advance |command| {
        let padding = " ".repeat(indent - command.cmd.len());
        io::println(fmt!("    %s%s%s",
                         command.cmd, padding, command.usage_line));
    }

    io::print(
        "\n\
        Use \"rust help <command>\" for more information about a command.\n\
        \n"
    );

}

pub fn main() {
    let os_args = os::args();

    if (os_args.len() > 1 && (os_args[1] == ~"-v" || os_args[1] == ~"--version")) {
        rustc::version(os_args[0]);
        unsafe { exit(0); }
    }

    let args = os_args.tail();

    if !args.is_empty() {
        let r = find_cmd(*args.head());
        for r.iter().advance |command| {
            let result = do_command(command, args.tail());
            match result {
                Valid(exit_code) => unsafe { exit(exit_code.to_i32()) },
                _                => loop
            }
        }
    }

    usage();
}
