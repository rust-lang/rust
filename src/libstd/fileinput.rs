// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
A convience device for iterating through the lines in a series of
files. Very similar to [the Python module of the same
name](http://docs.python.org/3.3/library/fileinput.html).

It allows the programmer to automatically take filenames from the
command line arguments (via `input` and `input_state`), as well as
specify them as a vector directly (`input_vec` and
`input_vec_state`). The files are opened as necessary, so any files
that can't be opened only cause an error when reached in the
iteration.

On the command line, `stdin` is represented by a filename of `-` (a
single hyphen) and in the functions that take a vector directly
(e.g. `input_vec`) it is represented by `None`. Note `stdin` is *not*
reset once it has been finished, so attempting to iterate on `[None,
None]` will only take input once unless `io::stdin().seek(0, SeekSet)`
is called between.

The `pathify` function handles converting a list of file paths as
strings to the appropriate format, including the (optional) conversion
of `"-"` to `stdin`.

# Basic

In many cases, one can use the `input_*` functions without having
to handle any `FileInput` structs. E.g. a simple `cat` program

    for input |line| {
        io::println(line)
    }

or a program that numbers lines after concatenating two files

    for input_vec_state(pathify([~"a.txt", ~"b.txt"])) |line, state| {
        io::println(fmt!("%u: %s", state.line_num,
                                   line));
    }

The 2 `_vec` functions take a vec of file names (and empty means
read from `stdin`), the other 2 use the command line arguments.

# Advanced

For more complicated uses (e.g. if one needs to pause iteration and
resume it later), a `FileInput` instance can be constructed via the
`from_vec`, `from_vec_raw` and `from_args` functions.

Once created, the `lines_each` and `lines_each_state` methods
allow one to iterate on the lines (the latter provides more
information about the position within the iteration to the caller.

It is possible (and safe) to skip lines and files using the
`read_line` and `next_file` methods.

E.g. the following (pointless) program reads until an empty line,
pauses for user input, skips the current file and then numbers the
remaining lines (where the numbers are from the start of the file,
rather than the total line count).

    let mut in = FileInput::from_vec(pathify([~"a.txt", ~"b.txt", ~"c.txt"],
                                             true));

    for in.lines_each |line| {
        if line.is_empty() {
            break
        }
        io::println(line);
    }

    io::println("Continue?");

    if io::stdin().read_line() == ~"yes" {
        in.next_file(); // skip!

        for in.lines_each_state |line, state| {
           io::println(fmt!("%u: %s", state.line_num_file,
                                      line))
        }
    }
*/

use core::prelude::*;
use core::io::ReaderUtil;

/**
A summary of the internal state of a FileInput object. `line_num` and
`line_num_file` represent the number of lines read in total and in the
current file respectively.
*/
pub struct FileInputState {
    current_path: Option<Path>,
    line_num: uint,
    line_num_file: uint
}

impl FileInputState {
    fn is_stdin(&self) -> bool {
        self.current_path.is_none()
    }

    fn is_first_line(&self) -> bool {
        self.line_num_file == 1
    }
}

priv struct FileInput {
    /**
    `Some(path)` is the file represented by `path`, `None` is
    `stdin`. Consumed as the files are read.
    */
    files: ~[Option<Path>],
    /**
    The current file: `Some(r)` for an open file, `None` before
    starting and after reading everything.
    */
    current_reader: Option<@io::Reader>,
    state: FileInputState
}

impl FileInput {
    /**
    Create a `FileInput` object from a vec of files. An empty
    vec means lines are read from `stdin` (use `from_vec_raw` to stop
    this behaviour). Any occurence of `None` represents `stdin`.
    */
    static pure fn from_vec(files: ~[Option<Path>]) -> FileInput {
        FileInput::from_vec_raw(
            if files.is_empty() {
                ~[None]
            } else {
                files
            })
    }

    /**
    Identical to `from_vec`, but an empty `files` vec stays
    empty. (`None` is `stdin`.)
    */
    static pure fn from_vec_raw(files: ~[Option<Path>])
                                         -> FileInput {
        FileInput {
            files: files,
            current_reader: None,
            state: FileInputState {
                current_path: None,
                line_num: 0,
                line_num_file: 0
            }
        }
    }

    /**
    Create a `FileInput` object from the command line
    arguments. `-` represents `stdin`.
    */
    static fn from_args() -> FileInput {
        let args = os::args(),
            pathed = pathify(args.tail(), true);
        FileInput::from_vec(pathed)
    }

    priv fn current_file_eof(&self) -> bool {
        match self.current_reader {
            None => false,
            Some(r) => r.eof()
        }
    }

    /**
    Skip to the next file in the queue. Can `fail` when opening
    a file.
    */
    pub fn next_file(&mut self) {
        // No more files
        if self.files.is_empty() {
            self.current_reader = None;
            return;
        }

        let path_option = self.files.shift(),
            file = match path_option {
                None => io::stdin(),
                Some(ref path) => io::file_reader(path).get()
            };

        self.current_reader = Some(file);
        self.state.current_path = path_option;
        self.state.line_num_file = 0;
    }

    /**
    Attempt to open the next file if there is none currently open,
    or if the current one is EOF'd.
    */
    priv fn next_file_if_eof(&mut self) {
        match self.current_reader {
            None => self.next_file(),
            Some(r) => {
                if r.eof() {
                    self.next_file()
                }
            }
        }
    }

    /**
    Read a single line. Returns `None` if there are no remaining lines
    in any remaining file. (Automatically opens files as required, see
    `next_file` for details.)

    (Name to avoid conflicting with `core::io::ReaderUtil::read_line`.)
    */
    pub fn next_line(&mut self) -> Option<~str> {
        loop {
            // iterate until there is a file that can be read from
            self.next_file_if_eof();
            match self.current_reader {
                None => {
                    // no file has any content
                    return None;
                },
                Some(r) => {
                    let l = r.read_line();

                    // at the end of this file, and we read nothing, so
                    // go to the next file
                    if r.eof() && l.is_empty() {
                        loop;
                    }
                    self.state.line_num += 1;
                    self.state.line_num_file += 1;
                    return Some(l);
                }
            }
        }
    }

    /**
    Call `f` on the lines in the files in succession, stopping if
    it ever returns `false`.

    State is preserved across calls.

    (The name is to avoid conflict with
    `core::io::ReaderUtil::each_line`.)
    */
    pub fn lines_each(&mut self, f: &fn(~str) -> bool) {
        loop {
            match self.next_line() {
                None => break,
                Some(line) => {
                    if !f(line) {
                        break;
                    }
                }
            }
        }
    }

    /**
    Apply `f` to each line successively, along with some state
    (line numbers and file names, see documentation for
    `FileInputState`). Otherwise identical to `lines_each`.
    */
    pub fn lines_each_state(&mut self,
                            f: &fn(~str, &FileInputState) -> bool) {
        loop {
            match self.next_line() {
                None => break,
                Some(line) => {
                    if !f(line, &self.state) {
                        break;
                    }
                }
            }
        }
    }
}

/**
Convert a list of strings to an appropriate form for a `FileInput`
instance. `stdin_hyphen` controls whether `-` represents `stdin` or
not.
*/
// XXX: stupid, unclear name
pub fn pathify(vec: &[~str], stdin_hyphen : bool) -> ~[Option<Path>] {
    vec::map(vec, |&str : & ~str| {
        if stdin_hyphen && str == ~"-" {
            None
        } else {
            Some(Path(str))
        }
    })
}

/**
Iterate directly over the command line arguments (no arguments implies
reading from `stdin`).

Fails when attempting to read from a file that can't be opened.
*/
pub fn input(f: &fn(~str) -> bool) {
    let mut i = FileInput::from_args();
    i.lines_each(f);
}

/**
Iterate directly over the command line arguments (no arguments
implies reading from `stdin`) with the current state of the iteration
provided at each call.

Fails when attempting to read from a file that can't be opened.
*/
pub fn input_state(f: &fn(~str, &FileInputState) -> bool) {
    let mut i = FileInput::from_args();
    i.lines_each_state(f);
}

/**
Iterate over a vec of files (an empty vec implies just `stdin`).

Fails when attempting to read from a file that can't be opened.
*/
pub fn input_vec(files: ~[Option<Path>], f: &fn(~str) -> bool) {
    let mut i = FileInput::from_vec(files);
    i.lines_each(f);
}

/**
Iterate over a vec of files (an empty vec implies just `stdin`) with
the current state of the iteration provided at each call.

Fails when attempting to read from a file that can't be opened.
*/
pub fn input_vec_state(files: ~[Option<Path>],
                       f: &fn(~str, &FileInputState) -> bool) {
    let mut i = FileInput::from_vec(files);
    i.lines_each_state(f);
}

#[cfg(test)]
mod test {
    use core::io::WriterUtil;
    use core::prelude::*;
    use super::{FileInput, pathify, input_vec, input_vec_state};

    fn make_file(path : &Path, contents: &[~str]) {
        let file = io::file_writer(path, [io::Create, io::Truncate]).get();

        for contents.each |&str| {
            file.write_str(str);
            file.write_char('\n');
        }
    }

    #[test]
    fn test_pathify() {
        let strs = [~"some/path",
                    ~"some/other/path"],
            paths = ~[Some(Path("some/path")),
                      Some(Path("some/other/path"))];

        fail_unless!(pathify(strs, true) == paths);
        fail_unless!(pathify(strs, false) == paths);

        fail_unless!(pathify([~"-"], true) == ~[None]);
        fail_unless!(pathify([~"-"], false) == ~[Some(Path("-"))]);
    }

    #[test]
    fn test_input_vec() {
        let mut all_lines = ~[];
        let filenames = pathify(vec::from_fn(
            3,
            |i| fmt!("tmp/lib-fileinput-test-input-vec-%u.tmp", i)), true);

        for filenames.eachi |i, &filename| {
            let contents =
                vec::from_fn(3, |j| fmt!("%u %u", i, j));
            make_file(&filename.get(), contents);
            all_lines.push_all(contents);
        }

        let mut read_lines = ~[];
        for input_vec(filenames) |line| {
            read_lines.push(line);
        }
        fail_unless!(read_lines == all_lines);
    }

    #[test]
    fn test_input_vec_state() {
        let filenames = pathify(vec::from_fn(
            3,
            |i|
            fmt!("tmp/lib-fileinput-test-input-vec-state-%u.tmp", i)),true);

        for filenames.eachi |i, &filename| {
            let contents =
                vec::from_fn(3, |j| fmt!("%u %u", i, j + 1));
            make_file(&filename.get(), contents);
        }

        for input_vec_state(filenames) |line, state| {
            let nums = str::split_char(line, ' ');

            let file_num = uint::from_str(nums[0]).get();
            let line_num = uint::from_str(nums[1]).get();

            fail_unless!(line_num == state.line_num_file);
            fail_unless!(file_num * 3 + line_num == state.line_num);
        }
    }

    #[test]
    fn test_next_file() {
        let filenames = pathify(vec::from_fn(
            3,
            |i|
            fmt!("tmp/lib-fileinput-test-next-file-%u.tmp", i)),true);

        for filenames.eachi |i, &filename| {
            let contents =
                vec::from_fn(3, |j| fmt!("%u %u", i, j + 1));
            make_file(&filename.get(), contents);
        }

        let mut in = FileInput::from_vec(filenames);

        // read once from 0
        fail_unless!(in.next_line() == Some(~"0 1"));
        in.next_file(); // skip the rest of 1

        // read all lines from 1 (but don't read any from 2),
        for uint::range(1, 4) |i| {
            fail_unless!(in.next_line() == Some(fmt!("1 %u", i)));
        }
        // 1 is finished, but 2 hasn't been started yet, so this will
        // just "skip" to the beginning of 2 (Python's fileinput does
        // the same)
        in.next_file();

        fail_unless!(in.next_line() == Some(~"2 1"));
    }

    #[test]
    #[should_fail]
    fn test_input_vec_missing_file() {
        for input_vec(pathify([~"this/file/doesnt/exist"], true)) |line| {
            io::println(line);
        }
    }
}
