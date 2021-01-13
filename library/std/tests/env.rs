use std::env::*;
use std::ffi::{OsStr, OsString};

use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};

fn make_rand_name() -> OsString {
    let rng = thread_rng();
    let n = format!("TEST{}", rng.sample_iter(&Alphanumeric).take(10).collect::<String>());
    let n = OsString::from(n);
    assert!(var_os(&n).is_none());
    n
}

fn eq(a: Option<OsString>, b: Option<&str>) {
    assert_eq!(a.as_ref().map(|s| &**s), b.map(OsStr::new).map(|s| &*s));
}

#[test]
fn test_set_var() {
    let n = make_rand_name();
    set_var(&n, "VALUE");
    eq(var_os(&n), Some("VALUE"));
}

#[test]
fn test_remove_var() {
    let n = make_rand_name();
    set_var(&n, "VALUE");
    remove_var(&n);
    eq(var_os(&n), None);
}

#[test]
fn test_set_var_overwrite() {
    let n = make_rand_name();
    set_var(&n, "1");
    set_var(&n, "2");
    eq(var_os(&n), Some("2"));
    set_var(&n, "");
    eq(var_os(&n), Some(""));
}

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn test_var_big() {
    let mut s = "".to_string();
    let mut i = 0;
    while i < 100 {
        s.push_str("aaaaaaaaaa");
        i += 1;
    }
    let n = make_rand_name();
    set_var(&n, &s);
    eq(var_os(&n), Some(&s));
}

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn test_env_set_get_huge() {
    let n = make_rand_name();
    let s = "x".repeat(10000);
    set_var(&n, &s);
    eq(var_os(&n), Some(&s));
    remove_var(&n);
    eq(var_os(&n), None);
}

#[test]
fn test_env_set_var() {
    let n = make_rand_name();

    let mut e = vars_os();
    set_var(&n, "VALUE");
    assert!(!e.any(|(k, v)| { &*k == &*n && &*v == "VALUE" }));

    assert!(vars_os().any(|(k, v)| { &*k == &*n && &*v == "VALUE" }));
}

#[test]
#[cfg_attr(not(any(unix, windows)), ignore, allow(unused))]
#[allow(deprecated)]
fn env_home_dir() {
    use std::path::PathBuf;

    fn var_to_os_string(var: Result<String, VarError>) -> Option<OsString> {
        match var {
            Ok(var) => Some(OsString::from(var)),
            Err(VarError::NotUnicode(var)) => Some(var),
            _ => None,
        }
    }

    cfg_if::cfg_if! {
        if #[cfg(unix)] {
            let oldhome = var_to_os_string(var("HOME"));

            set_var("HOME", "/home/MountainView");
            assert_eq!(home_dir(), Some(PathBuf::from("/home/MountainView")));

            remove_var("HOME");
            if cfg!(target_os = "android") {
                assert!(home_dir().is_none());
            } else {
                // When HOME is not set, some platforms return `None`,
                // but others return `Some` with a default.
                // Just check that it is not "/home/MountainView".
                assert_ne!(home_dir(), Some(PathBuf::from("/home/MountainView")));
            }

            if let Some(oldhome) = oldhome { set_var("HOME", oldhome); }
        } else if #[cfg(windows)] {
            let oldhome = var_to_os_string(var("HOME"));
            let olduserprofile = var_to_os_string(var("USERPROFILE"));

            remove_var("HOME");
            remove_var("USERPROFILE");

            assert!(home_dir().is_some());

            set_var("HOME", "/home/MountainView");
            assert_eq!(home_dir(), Some(PathBuf::from("/home/MountainView")));

            remove_var("HOME");

            set_var("USERPROFILE", "/home/MountainView");
            assert_eq!(home_dir(), Some(PathBuf::from("/home/MountainView")));

            set_var("HOME", "/home/MountainView");
            set_var("USERPROFILE", "/home/PaloAlto");
            assert_eq!(home_dir(), Some(PathBuf::from("/home/MountainView")));

            remove_var("HOME");
            remove_var("USERPROFILE");

            if let Some(oldhome) = oldhome { set_var("HOME", oldhome); }
            if let Some(olduserprofile) = olduserprofile { set_var("USERPROFILE", olduserprofile); }
        }
    }
}
