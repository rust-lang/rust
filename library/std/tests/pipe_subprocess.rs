fn main() {
    #[cfg(all(not(miri), any(unix, windows), not(target_os = "emscripten")))]
    {
        use std::io::{Read, pipe};
        use std::{env, process};

        if env::var("I_AM_THE_CHILD").is_ok() {
            child();
        } else {
            parent();
        }

        fn parent() {
            let me = env::current_exe().unwrap();
            // If `runner` is set up for current target, we'll be executing `./runner ./test`, not
            // just `./test`. For such a case, use the same arguments for child to avoid executing
            // `runner` without actual executable.
            let args = env::args();

            let (rx, tx) = pipe().unwrap();
            assert!(
                process::Command::new(me)
                    .args(args)
                    .env("I_AM_THE_CHILD", "1")
                    .stdout(tx)
                    .status()
                    .unwrap()
                    .success()
            );

            let mut s = String::new();
            (&rx).read_to_string(&mut s).unwrap();
            drop(rx);
            assert_eq!(s, "Heloo,\n");

            println!("Test pipe_subprocess.rs success");
        }

        fn child() {
            println!("Heloo,");
        }
    }
}
