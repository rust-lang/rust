// run-pass
// no-prefer-dynamic

// The test copies itself into a subdirectory with a non-ASCII name and then
// runs it as a child process within the subdirectory.  The parent process
// also adds an environment variable and an argument, both containing
// non-ASCII characters.  The child process ensures all the strings are
// intact.

// ignore-cloudabi no processes
// ignore-emscripten no processes
// ignore-sgx no processes

use std::io::prelude::*;
use std::io;
use std::fs;
use std::process::Command;
use std::env;
use std::path::Path;

fn main() {
    let my_args = env::args().collect::<Vec<_>>();
    let my_cwd  = env::current_dir().unwrap();
    let my_env  = env::vars().collect::<Vec<_>>();
    let my_path = env::current_exe().unwrap();
    let my_dir  = my_path.parent().unwrap();
    let my_ext  = my_path.extension().and_then(|s| s.to_str()).unwrap_or("");

    // some non-ASCII characters
    let blah       = "\u{3c0}\u{42f}\u{97f3}\u{e6}\u{221e}";

    let child_name = "child";
    let child_dir  = format!("process-spawn-with-unicode-params-{}", blah);

    // parameters sent to child / expected to be received from parent
    let arg = blah;
    let cwd = my_dir.join(&child_dir);
    let env = ("RUST_TEST_PROC_SPAWN_UNICODE".to_string(), blah.to_string());

    // am I the parent or the child?
    if my_args.len() == 1 {             // parent

        let child_filestem = Path::new(child_name);
        let child_filename = child_filestem.with_extension(my_ext);
        let child_path     = cwd.join(&child_filename);

        // make a separate directory for the child
        let _ = fs::create_dir(&cwd);
        fs::copy(&my_path, &child_path).unwrap();

        // run child
        let p = Command::new(&child_path)
                        .arg(arg)
                        .current_dir(&cwd)
                        .env(&env.0, &env.1)
                        .spawn().unwrap().wait_with_output().unwrap();

        // display the output
        io::stdout().write_all(&p.stdout).unwrap();
        io::stderr().write_all(&p.stderr).unwrap();

        // make sure the child succeeded
        assert!(p.status.success());

    } else {                            // child

        // check working directory (don't try to compare with `cwd` here!)
        assert!(my_cwd.ends_with(&child_dir));

        // check arguments
        assert_eq!(&*my_args[1], arg);

        // check environment variable
        assert!(my_env.contains(&env));

    };
}
