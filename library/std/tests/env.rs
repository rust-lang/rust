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
    // FIXME(skippy) there's no fix for deprecated_safe until tests can be run single threaded
    #[cfg_attr(not(bootstrap), allow(deprecated_safe))]
    set_var(&n, "VALUE");
    eq(var_os(&n), Some("VALUE"));
}

#[test]
fn test_remove_var() {
    let n = make_rand_name();
    // FIXME(skippy) there's no fix for deprecated_safe until tests can be run single threaded
    #[cfg_attr(not(bootstrap), allow(deprecated_safe))]
    set_var(&n, "VALUE");
    // FIXME(skippy) there's no fix for deprecated_safe until tests can be run single threaded
    #[cfg_attr(not(bootstrap), allow(deprecated_safe))]
    remove_var(&n);
    eq(var_os(&n), None);
}

#[test]
fn test_set_var_overwrite() {
    let n = make_rand_name();
    // FIXME(skippy) there's no fix for deprecated_safe until tests can be run single threaded
    #[cfg_attr(not(bootstrap), allow(deprecated_safe))]
    set_var(&n, "1");
    // FIXME(skippy) there's no fix for deprecated_safe until tests can be run single threaded
    #[cfg_attr(not(bootstrap), allow(deprecated_safe))]
    set_var(&n, "2");
    eq(var_os(&n), Some("2"));
    // FIXME(skippy) there's no fix for deprecated_safe until tests can be run single threaded
    #[cfg_attr(not(bootstrap), allow(deprecated_safe))]
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
    // FIXME(skippy) there's no fix for deprecated_safe until tests can be run single threaded
    #[cfg_attr(not(bootstrap), allow(deprecated_safe))]
    set_var(&n, &s);
    eq(var_os(&n), Some(&s));
}

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn test_env_set_get_huge() {
    let n = make_rand_name();
    let s = "x".repeat(10000);
    // FIXME(skippy) there's no fix for deprecated_safe until tests can be run single threaded
    #[cfg_attr(not(bootstrap), allow(deprecated_safe))]
    set_var(&n, &s);
    eq(var_os(&n), Some(&s));
    // FIXME(skippy) there's no fix for deprecated_safe until tests can be run single threaded
    #[cfg_attr(not(bootstrap), allow(deprecated_safe))]
    remove_var(&n);
    eq(var_os(&n), None);
}

#[test]
fn test_env_set_var() {
    let n = make_rand_name();

    let mut e = vars_os();
    // FIXME(skippy) there's no fix for deprecated_safe until tests can be run single threaded
    #[cfg_attr(not(bootstrap), allow(deprecated_safe))]
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

            // FIXME(skippy) there's no fix for deprecated_safe until tests can be run single threaded
            #[cfg_attr(not(bootstrap), allow(deprecated_safe))]
            set_var("HOME", "/home/MountainView");
            assert_eq!(home_dir(), Some(PathBuf::from("/home/MountainView")));

            // FIXME(skippy) there's no fix for deprecated_safe until tests can be run single threaded
            #[cfg_attr(not(bootstrap), allow(deprecated_safe))]
            remove_var("HOME");
            if cfg!(target_os = "android") {
                assert!(home_dir().is_none());
            } else {
                // When HOME is not set, some platforms return `None`,
                // but others return `Some` with a default.
                // Just check that it is not "/home/MountainView".
                assert_ne!(home_dir(), Some(PathBuf::from("/home/MountainView")));
            }

            // FIXME(skippy) there's no fix for deprecated_safe until tests can be run single threaded
            #[cfg_attr(not(bootstrap), allow(deprecated_safe))]
            if let Some(oldhome) = oldhome { set_var("HOME", oldhome); }
        } else if #[cfg(windows)] {
            let oldhome = var_to_os_string(var("HOME"));
            let olduserprofile = var_to_os_string(var("USERPROFILE"));

            // SAFETY: inside a cfg!(windows) section, remove_var is always sound
            #[cfg_attr(bootstrap, allow(unused_unsafe))]
            unsafe {
                remove_var("HOME");
                remove_var("USERPROFILE");
            }

            assert!(home_dir().is_some());

            set_var("HOME", "/home/MountainView");
            assert_eq!(home_dir(), Some(PathBuf::from("/home/MountainView")));

            // SAFETY: inside a cfg!(windows) section, remove_var is always sound
            #[cfg_attr(bootstrap, allow(unused_unsafe))]
            unsafe {
                remove_var("HOME");
            }

            // SAFETY: inside a cfg!(windows) section, set_var is always sound
            #[cfg_attr(bootstrap, allow(unused_unsafe))]
            unsafe {
                set_var("USERPROFILE", "/home/MountainView");
            }
            assert_eq!(home_dir(), Some(PathBuf::from("/home/MountainView")));

            // SAFETY: inside a cfg!(windows) section, set_var is always sound
            #[cfg_attr(bootstrap, allow(unused_unsafe))]
            unsafe {
                set_var("HOME", "/home/MountainView");
                set_var("USERPROFILE", "/home/PaloAlto");
            }
            assert_eq!(home_dir(), Some(PathBuf::from("/home/MountainView")));

            // SAFETY: inside a cfg!(windows) section, remove_var is always sound
            #[cfg_attr(bootstrap, allow(unused_unsafe))]
            unsafe {
                remove_var("HOME");
                remove_var("USERPROFILE");
            }

            // SAFETY: inside a cfg!(windows) section, set_var is always sound
            #[cfg_attr(bootstrap, allow(unused_unsafe))]
            unsafe {
                if let Some(oldhome) = oldhome { set_var("HOME", oldhome); }
                if let Some(olduserprofile) = olduserprofile { set_var("USERPROFILE", olduserprofile); }
            }
        }
    }
}
