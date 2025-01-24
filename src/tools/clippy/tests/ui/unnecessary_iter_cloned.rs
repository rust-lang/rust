#![allow(unused_assignments)]
#![warn(clippy::unnecessary_to_owned)]

#[allow(dead_code)]
#[derive(Clone, Copy)]
enum FileType {
    Account,
    PrivateKey,
    Certificate,
}

fn main() {
    let path = std::path::Path::new("x");

    let _ = check_files(&[(FileType::Account, path)]);
    let _ = check_files_vec(vec![(FileType::Account, path)]);

    // negative tests
    let _ = check_files_ref(&[(FileType::Account, path)]);
    let _ = check_files_mut(&[(FileType::Account, path)]);
    let _ = check_files_ref_mut(&[(FileType::Account, path)]);
    let _ = check_files_self_and_arg(&[(FileType::Account, path)]);
    let _ = check_files_mut_path_buf(&[(FileType::Account, std::path::PathBuf::new())]);

    check_mut_iteratee_and_modify_inner_variable();
}

// `check_files` and its variants are based on:
// https://github.com/breard-r/acmed/blob/1f0dcc32aadbc5e52de6d23b9703554c0f925113/acmed/src/storage.rs#L262
fn check_files(files: &[(FileType, &std::path::Path)]) -> bool {
    for (t, path) in files.iter().copied() {
        let other = match get_file_path(&t) {
            Ok(p) => p,
            Err(_) => {
                return false;
            },
        };
        if !path.is_file() || !other.is_file() {
            return false;
        }
    }
    true
}

fn check_files_vec(files: Vec<(FileType, &std::path::Path)>) -> bool {
    for (t, path) in files.iter().copied() {
        let other = match get_file_path(&t) {
            Ok(p) => p,
            Err(_) => {
                return false;
            },
        };
        if !path.is_file() || !other.is_file() {
            return false;
        }
    }
    true
}

fn check_files_ref(files: &[(FileType, &std::path::Path)]) -> bool {
    for (ref t, path) in files.iter().copied() {
        let other = match get_file_path(t) {
            Ok(p) => p,
            Err(_) => {
                return false;
            },
        };
        if !path.is_file() || !other.is_file() {
            return false;
        }
    }
    true
}

#[allow(unused_assignments)]
fn check_files_mut(files: &[(FileType, &std::path::Path)]) -> bool {
    for (mut t, path) in files.iter().copied() {
        t = FileType::PrivateKey;
        let other = match get_file_path(&t) {
            Ok(p) => p,
            Err(_) => {
                return false;
            },
        };
        if !path.is_file() || !other.is_file() {
            return false;
        }
    }
    true
}

fn check_files_ref_mut(files: &[(FileType, &std::path::Path)]) -> bool {
    for (ref mut t, path) in files.iter().copied() {
        *t = FileType::PrivateKey;
        let other = match get_file_path(t) {
            Ok(p) => p,
            Err(_) => {
                return false;
            },
        };
        if !path.is_file() || !other.is_file() {
            return false;
        }
    }
    true
}

fn check_files_self_and_arg(files: &[(FileType, &std::path::Path)]) -> bool {
    for (t, path) in files.iter().copied() {
        let other = match get_file_path(&t) {
            Ok(p) => p,
            Err(_) => {
                return false;
            },
        };
        if !path.join(path).is_file() || !other.is_file() {
            return false;
        }
    }
    true
}

#[allow(unused_assignments)]
fn check_files_mut_path_buf(files: &[(FileType, std::path::PathBuf)]) -> bool {
    for (mut t, path) in files.iter().cloned() {
        t = FileType::PrivateKey;
        let other = match get_file_path(&t) {
            Ok(p) => p,
            Err(_) => {
                return false;
            },
        };
        if !path.is_file() || !other.is_file() {
            return false;
        }
    }
    true
}

fn get_file_path(_file_type: &FileType) -> Result<std::path::PathBuf, std::io::Error> {
    Ok(std::path::PathBuf::new())
}

// Issue 12098
// https://github.com/rust-lang/rust-clippy/issues/12098
// no message emits
fn check_mut_iteratee_and_modify_inner_variable() {
    struct Test {
        list: Vec<String>,
        mut_this: bool,
    }

    impl Test {
        fn list(&self) -> &[String] {
            &self.list
        }
    }

    let mut test = Test {
        list: vec![String::from("foo"), String::from("bar")],
        mut_this: false,
    };

    for _item in test.list().to_vec() {
        println!("{}", _item);

        test.mut_this = true;
        {
            test.mut_this = true;
        }
    }
}

mod issue_12821 {
    fn foo() {
        let v: Vec<_> = "hello".chars().collect();
        for c in v.iter().cloned() {
            //~^ ERROR: unnecessary use of `cloned`
            println!("{c}"); // should not suggest to remove `&`
        }
    }

    fn bar() {
        let v: Vec<_> = "hello".chars().collect();
        for c in v.iter().cloned() {
            //~^ ERROR: unnecessary use of `cloned`
            let ref_c = &c; //~ HELP: remove any references to the binding
            println!("{ref_c}");
        }
    }

    fn baz() {
        let v: Vec<_> = "hello".chars().enumerate().collect();
        for (i, c) in v.iter().cloned() {
            //~^ ERROR: unnecessary use of `cloned`
            let ref_c = &c; //~ HELP: remove any references to the binding
            let ref_i = &i;
            println!("{i} {ref_c}"); // should not suggest to remove `&` from `i`
        }
    }
}
