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
A library for iterating through the lines in a series of
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

The `make_path_option_vec` function handles converting a list of file paths as
strings to the appropriate format, including the (optional) conversion
of `"-"` to `stdin`.

# Basic

In many cases, one can use the `input_*` functions without having
to handle any `FileInput` structs. E.g. a simple `cat` program

    for input |line| {
        io::println(line)
    }

or a program that numbers lines after concatenating two files

    for input_vec_state(make_path_option_vec([~"a.txt", ~"b.txt"])) |line, state| {
        io::println(format!("{}: %s", state.line_num,
                                   line));
    }

The two `input_vec*` functions take a vec of file names (where empty
means read from `stdin`), the other two functions use the command line
arguments.

# Advanced

For more complicated uses (e.g. if one needs to pause iteration and
resume it later), a `FileInput` instance can be constructed via the
`from_vec`, `from_vec_raw` and `from_args` functions.

Once created, the `each_line` (from the `std::io::ReaderUtil` trait)
and `each_line_state` methods allow one to iterate on the lines; the
latter provides more information about the position within the
iteration to the caller.

It is possible (and safe) to skip lines and files using the
`read_line` and `next_file` methods. Also, `FileInput` implements
`std::io::Reader`, and the state will be updated correctly while
using any of those methods.

E.g. the following program reads until an empty line, pauses for user
input, skips the current file and then numbers the remaining lines
(where the numbers are from the start of each file, rather than the
total line count).

    let input = FileInput::from_vec(pathify([~"a.txt", ~"b.txt", ~"c.txt"],
                                             true));

    for input.each_line |line| {
        if line.is_empty() {
            break
        }
        io::println(line);
    }

    io::println("Continue?");

    if io::stdin().read_line() == ~"yes" {
        input.next_file(); // skip!

        for input.each_line_state |line, state| {
           io::println(format!("{}: %s", state.line_num_file,
                                      line))
        }
    }
*/

#[allow(missing_doc)];


use std::io::ReaderUtil;
use std::io;
use std::os;

/**
A summary of the internal state of a `FileInput` object. `line_num`
and `line_num_file` represent the number of lines read in total and in
the current file respectively. `current_path` is `None` if the current
file is `stdin`.
*/
#[deriving(Clone)]
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

struct FileInput_ {
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
    state: FileInputState,

    /**
    Used to keep track of whether we need to insert the newline at the
    end of a file that is missing it, which is needed to separate the
    last and first lines.
    */
    previous_was_newline: bool
}


// FIXME #5723: remove this when Reader has &mut self.
// Removing it would mean giving read_byte in the Reader impl for
// FileInput &mut self, which in turn means giving most of the
// io::Reader trait methods &mut self. That can't be done right now
// because of io::with_bytes_reader and #5723.
// Should be removable via
// "self.fi" -> "self." and renaming FileInput_. Documentation above
// will likely have to be updated to use `let mut in = ...`.
pub struct FileInput  {
    fi: @mut FileInput_
}

impl FileInput {
    /**
    Create a `FileInput` object from a vec of files. An empty
    vec means lines are read from `stdin` (use `from_vec_raw` to stop
    this behaviour). Any occurrence of `None` represents `stdin`.
    */
    pub fn from_vec(files: ~[Option<Path>]) -> FileInput {
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
    pub fn from_vec_raw(files: ~[Option<Path>])
                                         -> FileInput {
        FileInput{
            fi: @mut FileInput_ {
                files: files,
                current_reader: None,
                state: FileInputState {
                    current_path: None,
                    line_num: 0,
                    line_num_file: 0
                },
                // there was no previous unended line
                previous_was_newline: true
            }
        }
    }

    /**
    Create a `FileInput` object from the command line
    arguments. `"-"` represents `stdin`.
    */
    pub fn from_args() -> FileInput {
        let args = os::args();
        let pathed = make_path_option_vec(args.tail(), true);
        FileInput::from_vec(pathed)
    }

    fn current_file_eof(&self) -> bool {
        match self.fi.current_reader {
            None => false,
            Some(r) => r.eof()
        }
    }

    /**
    Skip to the next file in the queue. Can `fail` when opening
    a file.

    Returns `false` if there is no more files, and `true` when it
    successfully opens the next file.
    */

    pub fn next_file(&self) -> bool {
        // No more files

        if self.fi.files.is_empty() {
            self.fi.current_reader = None;
            return false;
        }

        let path_option = self.fi.files.shift();
        let file = match path_option {
            None => io::stdin(),
            Some(ref path) => io::file_reader(path).unwrap()
        };

        self.fi.current_reader = Some(file);
        self.fi.state.current_path = path_option;
        self.fi.state.line_num_file = 0;
        true
    }

    /**
    Attempt to open the next file if there is none currently open,
    or if the current one is EOF'd.

    Returns `true` if it had to move to the next file and did
    so successfully.
    */
    fn next_file_if_eof(&self) -> bool {
        match self.fi.current_reader {
            None => self.next_file(),
            Some(r) => {
                if r.eof() {
                    self.next_file()
                } else {
                    false
                }
            }
        }
    }

    /**
    Apply `f` to each line successively, along with some state
    (line numbers and file names, see documentation for
    `FileInputState`). Otherwise identical to `lines_each`.
    */
    pub fn each_line_state(&self,
                            f: &fn(&str, FileInputState) -> bool) -> bool {
         self.each_line(|line| f(line, self.fi.state.clone()))
    }


    /**
    Retrieve the current `FileInputState` information.
    */
    pub fn state(&self) -> FileInputState {
        self.fi.state.clone()
    }
}

impl io::Reader for FileInput {
    fn read_byte(&self) -> int {
        loop {
            let stepped = self.next_file_if_eof();

            // if we moved to the next file, and the previous
            // character wasn't \n, then there is an unfinished line
            // from the previous file. This library models
            // line-by-line processing and the trailing line of the
            // previous file and the leading of the current file
            // should be considered different, so we need to insert a
            // fake line separator
            if stepped && !self.fi.previous_was_newline {
                self.fi.state.line_num += 1;
                self.fi.state.line_num_file += 1;
                self.fi.previous_was_newline = true;
                return '\n' as int;
            }

            match self.fi.current_reader {
                None => return -1,
                Some(r) => {
                    let b = r.read_byte();

                    if b < 0 {
                        continue;
                    }

                    if b == '\n' as int {
                        self.fi.state.line_num += 1;
                        self.fi.state.line_num_file += 1;
                        self.fi.previous_was_newline = true;
                    } else {
                        self.fi.previous_was_newline = false;
                    }

                    return b;
                }
            }
        }
    }
    fn read(&self, buf: &mut [u8], len: uint) -> uint {
        let mut count = 0;
        while count < len {
            let b = self.read_byte();
            if b < 0 { break }

            buf[count] = b as u8;
            count += 1;
        }

        count
    }
    fn eof(&self) -> bool {
        // we've run out of files, and current_reader is either None or eof.

        self.fi.files.is_empty() &&
            match self.fi.current_reader { None => true, Some(r) => r.eof() }

    }
    fn seek(&self, offset: int, whence: io::SeekStyle) {
        match self.fi.current_reader {
            None => {},
            Some(r) => r.seek(offset, whence)
        }
    }
    fn tell(&self) -> uint {
        match self.fi.current_reader {
            None => 0,
            Some(r) => r.tell()
        }
    }
}

/**
Convert a list of strings to an appropriate form for a `FileInput`
instance. `stdin_hyphen` controls whether `-` represents `stdin` or
a literal `-`.
*/
pub fn make_path_option_vec(vec: &[~str], stdin_hyphen : bool) -> ~[Option<Path>] {
    vec.iter().map(|str| {
        if stdin_hyphen && "-" == *str {
            None
        } else {
            Some(Path(*str))
        }
    }).collect()
}

/**
Iterate directly over the command line arguments (no arguments implies
reading from `stdin`).

Fails when attempting to read from a file that can't be opened.
*/
pub fn input(f: &fn(&str) -> bool) -> bool {
    let i = FileInput::from_args();
    i.each_line(f)
}

/**
Iterate directly over the command line arguments (no arguments
implies reading from `stdin`) with the current state of the iteration
provided at each call.

Fails when attempting to read from a file that can't be opened.
*/
pub fn input_state(f: &fn(&str, FileInputState) -> bool) -> bool {
    let i = FileInput::from_args();
    i.each_line_state(f)
}

/**
Iterate over a vector of files (an empty vector implies just `stdin`).

Fails when attempting to read from a file that can't be opened.
*/
pub fn input_vec(files: ~[Option<Path>], f: &fn(&str) -> bool) -> bool {
    let i = FileInput::from_vec(files);
    i.each_line(f)
}

/**
Iterate over a vector of files (an empty vector implies just `stdin`)
with the current state of the iteration provided at each call.

Fails when attempting to read from a file that can't be opened.
*/
pub fn input_vec_state(files: ~[Option<Path>],
                       f: &fn(&str, FileInputState) -> bool) -> bool {
    let i = FileInput::from_vec(files);
    i.each_line_state(f)
}

#[cfg(test)]
mod test {

    use super::{FileInput, make_path_option_vec, input_vec, input_vec_state};

    use std::rt::io;
    use std::rt::io::Writer;
    use std::rt::io::file;
    use std::vec;

    fn make_file(path : &Path, contents: &[~str]) {
        let mut file = file::open(path, io::CreateOrTruncate, io::Write).unwrap();

        for str in contents.iter() {
            file.write(str.as_bytes());
            file.write(['\n' as u8]);
        }
    }

    #[test]
    fn test_make_path_option_vec() {
        let strs = [~"some/path",
                    ~"some/other/path"];
        let paths = ~[Some(Path("some/path")),
                      Some(Path("some/other/path"))];

        assert_eq!(make_path_option_vec(strs, true), paths.clone());
        assert_eq!(make_path_option_vec(strs, false), paths);

        assert_eq!(make_path_option_vec([~"-"], true), ~[None]);
        assert_eq!(make_path_option_vec([~"-"], false), ~[Some(Path("-"))]);
    }

    #[test]
    fn test_fileinput_read_byte() {
        let filenames = make_path_option_vec(vec::from_fn(
            3,
            |i| format!("tmp/lib-fileinput-test-fileinput-read-byte-{}.tmp", i)), true);

        // 3 files containing 0\n, 1\n, and 2\n respectively
        for (i, filename) in filenames.iter().enumerate() {
            make_file(filename.get_ref(), [format!("{}", i)]);
        }

        let fi = FileInput::from_vec(filenames.clone());

        for (line, c) in "012".iter().enumerate() {
            assert_eq!(fi.read_byte(), c as int);
            assert_eq!(fi.state().line_num, line);
            assert_eq!(fi.state().line_num_file, 0);
            assert_eq!(fi.read_byte(), '\n' as int);
            assert_eq!(fi.state().line_num, line + 1);
            assert_eq!(fi.state().line_num_file, 1);

            assert_eq!(fi.state().current_path.clone(), filenames[line].clone());
        }

        assert_eq!(fi.read_byte(), -1);
        assert!(fi.eof());
        assert_eq!(fi.state().line_num, 3)

    }

    #[test]
    fn test_fileinput_read() {
        let filenames = make_path_option_vec(vec::from_fn(
            3,
            |i| format!("tmp/lib-fileinput-test-fileinput-read-{}.tmp", i)), true);

        // 3 files containing 1\n, 2\n, and 3\n respectively
        for (i, filename) in filenames.iter().enumerate() {
            make_file(filename.get_ref(), [format!("{}", i)]);
        }

        let fi = FileInput::from_vec(filenames);
        let mut buf : ~[u8] = vec::from_elem(6, 0u8);
        let count = fi.read(buf, 10);
        assert_eq!(count, 6);
        assert_eq!(buf, "0\n1\n2\n".as_bytes().to_owned());
        assert!(fi.eof())
        assert_eq!(fi.state().line_num, 3);
    }

    #[test]
    fn test_input_vec() {
        let mut all_lines = ~[];
        let filenames = make_path_option_vec(vec::from_fn(
            3,
            |i| format!("tmp/lib-fileinput-test-input-vec-{}.tmp", i)), true);

        for (i, filename) in filenames.iter().enumerate() {
            let contents =
                vec::from_fn(3, |j| format!("{} {}", i, j));
            make_file(filename.get_ref(), contents);
            debug2!("contents={:?}", contents);
            all_lines.push_all(contents);
        }

        let mut read_lines = ~[];
        do input_vec(filenames) |line| {
            read_lines.push(line.to_owned());
            true
        };
        assert_eq!(read_lines, all_lines);
    }

    #[test]
    fn test_input_vec_state() {
        let filenames = make_path_option_vec(vec::from_fn(
            3,
            |i| format!("tmp/lib-fileinput-test-input-vec-state-{}.tmp", i)),true);

        for (i, filename) in filenames.iter().enumerate() {
            let contents =
                vec::from_fn(3, |j| format!("{} {}", i, j + 1));
            make_file(filename.get_ref(), contents);
        }

        do input_vec_state(filenames) |line, state| {
            let nums: ~[&str] = line.split_iter(' ').collect();
            let file_num = from_str::<uint>(nums[0]).unwrap();
            let line_num = from_str::<uint>(nums[1]).unwrap();
            assert_eq!(line_num, state.line_num_file);
            assert_eq!(file_num * 3 + line_num, state.line_num);
            true
        };
    }

    #[test]
    fn test_empty_files() {
        let filenames = make_path_option_vec(vec::from_fn(
            3,
            |i| format!("tmp/lib-fileinput-test-empty-files-{}.tmp", i)),true);

        make_file(filenames[0].get_ref(), [~"1", ~"2"]);
        make_file(filenames[1].get_ref(), []);
        make_file(filenames[2].get_ref(), [~"3", ~"4"]);

        let mut count = 0;
        do input_vec_state(filenames.clone()) |line, state| {
            let expected_path = match line {
                "1" | "2" => filenames[0].clone(),
                "3" | "4" => filenames[2].clone(),
                _ => fail2!("unexpected line")
            };
            assert_eq!(state.current_path.clone(), expected_path);
            count += 1;
            true
        };
        assert_eq!(count, 4);
    }

    #[test]
    fn test_no_trailing_newline() {
        let f1 =
            Some(Path("tmp/lib-fileinput-test-no-trailing-newline-1.tmp"));
        let f2 =
            Some(Path("tmp/lib-fileinput-test-no-trailing-newline-2.tmp"));

        {
            let mut wr = file::open(f1.get_ref(), io::CreateOrTruncate,
                                    io::Write).unwrap();
            wr.write("1\n2".as_bytes());
            let mut wr = file::open(f2.get_ref(), io::CreateOrTruncate,
                                    io::Write).unwrap();
            wr.write("3\n4".as_bytes());
        }

        let mut lines = ~[];
        do input_vec(~[f1, f2]) |line| {
            lines.push(line.to_owned());
            true
        };
        assert_eq!(lines, ~[~"1", ~"2", ~"3", ~"4"]);
    }


    #[test]
    fn test_next_file() {
        let filenames = make_path_option_vec(vec::from_fn(
            3,
            |i| format!("tmp/lib-fileinput-test-next-file-{}.tmp", i)),true);

        for (i, filename) in filenames.iter().enumerate() {
            let contents = vec::from_fn(3, |j| format!("{} {}", i, j + 1));
            make_file(filename.get_ref(), contents);
        }

        let input = FileInput::from_vec(filenames);

        // read once from 0
        assert_eq!(input.read_line(), ~"0 1");
        input.next_file(); // skip the rest of 1

        // read all lines from 1 (but don't read any from 2),
        for i in range(1u, 4) {
            assert_eq!(input.read_line(), format!("1 {}", i));
        }
        // 1 is finished, but 2 hasn't been started yet, so this will
        // just "skip" to the beginning of 2 (Python's fileinput does
        // the same)
        input.next_file();

        assert_eq!(input.read_line(), ~"2 1");
    }

    #[test]
    #[should_fail]
    fn test_input_vec_missing_file() {
        do input_vec(make_path_option_vec([~"this/file/doesnt/exist"], true)) |line| {
            println(line);
            true
        };
    }
}
