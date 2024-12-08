#![warn(clippy::zombie_processes)]
#![allow(clippy::if_same_then_else, clippy::ifs_same_cond)]

use std::process::{Child, Command};

fn main() {
    {
        // Check that #[expect] works
        #[expect(clippy::zombie_processes)]
        let mut x = Command::new("").spawn().unwrap();
    }

    {
        let mut x = Command::new("").spawn().unwrap();
        //~^ ERROR: spawned process is never `wait()`ed on
        x.kill();
        x.id();
    }
    {
        let mut x = Command::new("").spawn().unwrap();
        x.wait().unwrap(); // OK
    }
    {
        let x = Command::new("").spawn().unwrap();
        x.wait_with_output().unwrap(); // OK
    }
    {
        let mut x = Command::new("").spawn().unwrap();
        x.try_wait().unwrap(); // OK
    }
    {
        let mut x = Command::new("").spawn().unwrap();
        let mut r = &mut x;
        r.wait().unwrap(); // OK, not calling `.wait()` directly on `x` but through `r` -> `x`
    }
    {
        let mut x = Command::new("").spawn().unwrap();
        process_child(x); // OK, other function might call `.wait()` so assume it does
    }
    {
        let mut x = Command::new("").spawn().unwrap();
        //~^ ERROR: spawned process is never `wait()`ed on
        let v = &x;
        // (allow shared refs is fine because one cannot call `.wait()` through that)
    }

    // https://github.com/rust-lang/rust-clippy/pull/11476#issuecomment-1718456033
    // Unconditionally exiting the process in various ways (should not lint)
    {
        let mut x = Command::new("").spawn().unwrap();
        std::process::exit(0);
    }
    {
        let mut x = Command::new("").spawn().unwrap();
        std::process::abort(); // same as above, but abort instead of exit
    }
    {
        let mut x = Command::new("").spawn().unwrap();
        if true { /* nothing */ }
        std::process::abort(); // still unconditionally exits
    }

    // Conditionally exiting
    // It should assume that it might not exit and still lint
    {
        let mut x = Command::new("").spawn().unwrap();
        //~^ ERROR: spawned process is never `wait()`ed on
        if true {
            std::process::exit(0);
        }
    }
    {
        let mut x = Command::new("").spawn().unwrap();
        //~^ ERROR: spawned process is never `wait()`ed on
        if true {
            while false {}
            // Calling `exit()` after leaving a while loop should still be linted.
            std::process::exit(0);
        }
    }

    {
        let mut x = { Command::new("").spawn().unwrap() };
        x.wait().unwrap();
    }

    {
        struct S {
            c: Child,
        }

        let mut s = S {
            c: Command::new("").spawn().unwrap(),
        };
        s.c.wait().unwrap();
    }

    {
        let mut x = Command::new("").spawn().unwrap();
        //~^ ERROR: spawned process is never `wait()`ed on
        if true {
            return;
        }
        x.wait().unwrap();
    }

    {
        let mut x = Command::new("").spawn().unwrap();
        //~^ ERROR: spawned process is never `wait()`ed on
        if true {
            x.wait().unwrap();
        }
    }

    {
        let mut x = Command::new("").spawn().unwrap();
        if true {
            x.wait().unwrap();
        } else if true {
            x.wait().unwrap();
        } else {
            x.wait().unwrap();
        }
    }

    {
        let mut x = Command::new("").spawn().unwrap();
        if true {
            x.wait().unwrap();
            return;
        }
        x.wait().unwrap();
    }

    {
        let mut x = Command::new("").spawn().unwrap();
        std::thread::spawn(move || {
            x.wait().unwrap();
        });
    }
}

fn process_child(c: Child) {
    todo!()
}
