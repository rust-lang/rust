// run-pass

#![allow(stable_features)]
// ignore-windows - this is a unix-specific test
// ignore-pretty issue #37199
// ignore-emscripten no processes
// ignore-sgx no processes

#![feature(process_exec)]

use std::env;
use std::os::unix::process::CommandExt;
use std::process::Command;

fn main() {
    let mut args = env::args();
    let me = args.next().unwrap();

    if let Some(arg) = args.next() {
        match &arg[..] {
            "test1" => println!("passed"),

            "exec-test1" => {
                let err = Command::new(&me).arg("test1").exec();
                panic!("failed to spawn: {}", err);
            }

            "exec-test2" => {
                Command::new("/path/to/nowhere").exec();
                println!("passed");
            }

            "exec-test3" => {
                Command::new(&me).arg("bad\0").exec();
                println!("passed");
            }

            "exec-test4" => {
                Command::new(&me).current_dir("/path/to/nowhere").exec();
                println!("passed");
            }

            "exec-test5" => {
                env::set_var("VARIABLE", "ABC");
                Command::new("definitely-not-a-real-binary").env("VARIABLE", "XYZ").exec();
                assert_eq!(env::var("VARIABLE").unwrap(), "ABC");
                println!("passed");
            }

            "exec-test6" => {
                let err = Command::new("echo").arg("passed").env_clear().exec();
                panic!("failed to spawn: {}", err);
            }

            "exec-test7" => {
                let err = Command::new("echo").arg("passed").env_remove("PATH").exec();
                panic!("failed to spawn: {}", err);
            }

            _ => panic!("unknown argument: {}", arg),
        }
        return
    }

    let output = Command::new(&me).arg("exec-test1").output().unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, b"passed\n");

    let output = Command::new(&me).arg("exec-test2").output().unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, b"passed\n");

    let output = Command::new(&me).arg("exec-test3").output().unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, b"passed\n");

    let output = Command::new(&me).arg("exec-test4").output().unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, b"passed\n");

    let output = Command::new(&me).arg("exec-test5").output().unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, b"passed\n");

    if cfg!(target_os = "linux") {
        let output = Command::new(&me).arg("exec-test6").output().unwrap();
        println!("{:?}", output);
        assert!(output.status.success());
        assert!(output.stderr.is_empty());
        assert_eq!(output.stdout, b"passed\n");

        let output = Command::new(&me).arg("exec-test7").output().unwrap();
        println!("{:?}", output);
        assert!(output.status.success());
        assert!(output.stderr.is_empty());
        assert_eq!(output.stdout, b"passed\n");
    }
}
