use super::*;

#[test]
#[cfg(unix)]
fn test_unix_absolute() {
    use crate::path::absolute;

    assert!(absolute("").is_err());

    let relative = "a/b";
    let mut expected = crate::env::current_dir().unwrap();
    expected.push(relative);
    assert_eq!(absolute(relative).unwrap().as_os_str(), expected.as_os_str());

    // Test how components are collected.
    assert_eq!(absolute("/a/b/c").unwrap().as_os_str(), Path::new("/a/b/c").as_os_str());
    assert_eq!(absolute("/a//b/c").unwrap().as_os_str(), Path::new("/a/b/c").as_os_str());
    assert_eq!(absolute("//a/b/c").unwrap().as_os_str(), Path::new("//a/b/c").as_os_str());
    assert_eq!(absolute("///a/b/c").unwrap().as_os_str(), Path::new("/a/b/c").as_os_str());
    assert_eq!(absolute("/a/b/c/").unwrap().as_os_str(), Path::new("/a/b/c/").as_os_str());
    assert_eq!(
        absolute("/a/./b/../c/.././..").unwrap().as_os_str(),
        Path::new("/a/b/../c/../..").as_os_str()
    );

    // Test leading `.` and `..` components
    let curdir = crate::env::current_dir().unwrap();
    assert_eq!(absolute("./a").unwrap().as_os_str(), curdir.join("a").as_os_str());
    assert_eq!(absolute("../a").unwrap().as_os_str(), curdir.join("../a").as_os_str());
    // return /pwd/../a
}

#[test]
#[cfg(windows)]
fn test_windows_absolute() {
    use crate::path::absolute;
    // An empty path is an error.
    assert!(absolute("").is_err());

    let relative = r"a\b";
    let mut expected = crate::env::current_dir().unwrap();
    expected.push(relative);
    assert_eq!(absolute(relative).unwrap().as_os_str(), expected.as_os_str());

    macro_rules! unchanged(
        ($path:expr) => {
            assert_eq!(absolute($path).unwrap().as_os_str(), Path::new($path).as_os_str());
        }
    );

    unchanged!(r"C:\path\to\file");
    unchanged!(r"C:\path\to\file\");
    unchanged!(r"\\server\share\to\file");
    unchanged!(r"\\server.\share.\to\file");
    unchanged!(r"\\.\PIPE\name");
    unchanged!(r"\\.\C:\path\to\COM1");
    unchanged!(r"\\?\C:\path\to\file");
    unchanged!(r"\\?\UNC\server\share\to\file");
    unchanged!(r"\\?\PIPE\name");
    // Verbatim paths are always unchanged, no matter what.
    unchanged!(r"\\?\path.\to/file..");

    assert_eq!(
        absolute(r"C:\path..\to.\file.").unwrap().as_os_str(),
        Path::new(r"C:\path..\to\file").as_os_str()
    );
    assert_eq!(absolute(r"COM1").unwrap().as_os_str(), Path::new(r"\\.\COM1").as_os_str());
}
