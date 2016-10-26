// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(test)]
mod test_find_program {
    use std::ffi::{OsStr, OsString};
    use std::process::Command;
    use std::collections::HashMap;
    use std::path::Path;
    use std::fs::canonicalize;
    use std::env::join_paths;

    fn gen_env() -> HashMap<OsString, OsString> {
        let env: HashMap<OsString, OsString> = HashMap::new();
        env.insert(OsString::from("HOMEDRIVE"), OsString::from("C:"));
        let p1 = canonicalize("./src/test/run-pass/process_command/fixtures/bin").unwrap();
        let p2 = canonicalize("./src/test/run-pass/process_command/fixtures").unwrap();
        let p3 = canonicalize("./src/test/run-pass/process_command").unwrap();
        let paths = vec![p1, p2, p3];
        let path = join_paths(paths).unwrap();
        env.insert(OsString::from("PATH"), OsString::from(&path));
        env.insert(OsString::from("USERNAME"), OsString::from("rust"));
        env
    }

    fn command_with_pathext(cmd: &str) -> Command {
        let mut env = gen_env();
        env.insert(
            OsString::from("PATHEXT"),
            OsString::from(".COM;.EXE;.BAT;.CMD;.VBS;.VBE;.MSC")
        );
        let mut cmd = Command::new(cmd);
        cmd.with_env(&env);
        cmd
    }

    fn command_without_pathext(cmd: &str) -> Command {
        let env = gen_env();
        let mut cmd = Command::new(cmd);
        cmd.with_env(&env);
        cmd
    }

    mod with_pathext_set {
        use super::command_with_pathext;

        fn command_on_path_found() {
            let c = command_with_pathext("bin");
            let bat = canonicalize("./src/test/run-pass/process_command/fixtures/bin/bin.bat");
            assert_eq!(bat.ok(), c.find_program());
        }

        fn command_not_found() {
            let c = command_with_pathext("missing");
            assert_eq!(None, c.find_program());
        }
    }

    mod without_pathext_set {
        use super::command_without_pathext;

        fn bat_command_not_found() {
            let c = command_without_pathext("bin");
            assert_eq!(None, c.find_program());
        }

        fn exe_command_found() {
            let c = command_without_pathext("exec");
            let exe = canonicalize("./src/test/run-pass/process_command/fixtures/bin/exec.exe");
            assert_eq!(exe.ok(), c.find_program());
        }
    }
}
