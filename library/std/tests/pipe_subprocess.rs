#![feature(anonymous_pipe)]

fn main() {
    #[cfg(all(not(miri), any(unix, windows)))]
    {
        use std::io::Read;
        use std::pipe::pipe;
        use std::{env, process};

        if env::var("I_AM_THE_CHILD").is_ok() {
            child();
        } else {
            parent();
        }

        fn parent() {
            let me = env::current_exe().unwrap();

            let (rx, tx) = pipe().unwrap();
            assert!(
                process::Command::new(me)
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
