// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic

// The test copies itself into a subdirectory with a non-ASCII name and then
// runs it as a child process within the subdirectory.  The parent process
// also adds an environment variable and an argument, both containing
// non-ASCII characters.  The child process ensures all the strings are
// intact.

use std::old_io;
use std::old_io::fs;
use std::old_io::Command;
use std::os;
use std::env;
use std::old_path::Path;

fn main() {
    let my_args = env::args().collect::<Vec<_>>();
    let my_cwd  = os::getcwd().unwrap();
    let my_env  = env::vars().collect::<Vec<_>>();
    let my_path = Path::new(os::self_exe_name().unwrap());
    let my_dir  = my_path.dir_path();
    let my_ext  = my_path.extension_str().unwrap_or("");

    // some non-ASCII characters
    let blah       = "\u03c0\u042f\u97f3\u00e6\u221e";

    let child_name = "child";
    let child_dir  = format!("process-spawn-with-unicode-params-{}", blah);

    // parameters sent to child / expected to be received from parent
    let arg = blah;
    let cwd = my_dir.join(Path::new(child_dir.clone()));
    let env = ("RUST_TEST_PROC_SPAWN_UNICODE".to_string(), blah.to_string());

    // am I the parent or the child?
    if my_args.len() == 1 {             // parent

        let child_filestem = Path::new(child_name);
        let child_filename = child_filestem.with_extension(my_ext);
        let child_path     = cwd.join(child_filename);

        // make a separate directory for the child
        drop(fs::mkdir(&cwd, old_io::USER_RWX).is_ok());
        assert!(fs::copy(&my_path, &child_path).is_ok());
        let mut my_env = my_env;
        my_env.push(env);

        // run child
        let p = Command::new(&child_path)
                        .arg(arg)
                        .cwd(&cwd)
                        .env_set_all(&my_env)
                        .spawn().unwrap().wait_with_output().unwrap();

        // display the output
        assert!(old_io::stdout().write(&p.output).is_ok());
        assert!(old_io::stderr().write(&p.error).is_ok());

        // make sure the child succeeded
        assert!(p.status.success());

    } else {                            // child

        // check working directory (don't try to compare with `cwd` here!)
        assert!(my_cwd.ends_with_path(&Path::new(child_dir)));

        // check arguments
        assert_eq!(&*my_args[1], arg);

        // check environment variable
        assert!(my_env.contains(&env));

    };
}
