use std::ffi::OsStr;
use std::path::Path;

#[test]
fn empty_path_ancestors() {
    let path = Path::new("");

    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(ancestors.next(), None);

    let mut rev_ancestors = path.ancestors();
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(ancestors.next(), None);
}

#[test]
fn curr_dir_only_path_ancestors() {
    let path = Path::new(".");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("."));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(ancestors.next(), None);

    let mut rev_ancestors = path.ancestors();
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("."));
    assert_eq!(rev_ancestors.next_back(), None);
}

#[test]
fn curr_dir_only_path_ancestors_rev() {
    let path = Path::new(".");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("."));
    // next_back() should only see "" leftover
    assert_eq!(ancestors.next_back().unwrap().as_os_str(), OsStr::new(""));
    // We have consumed "." and "", we should only observe
    // `None` being returned from either end
    assert_eq!(ancestors.next(), None);
    assert_eq!(ancestors.next_back(), None);

    // operates like next_back()
    let mut rev_ancestors = path.ancestors().rev();
    assert_eq!(rev_ancestors.next().unwrap().as_os_str(), OsStr::new(""));

    // operates like next()
    let mut rev_ancestors = rev_ancestors.rev();
    assert_eq!(rev_ancestors.next().unwrap().as_os_str(), OsStr::new("."));

    // fully consumed, should return None
    let mut rev_ancestors = rev_ancestors.rev();
    assert_eq!(rev_ancestors.next_back(), None);
}

#[test]
fn curr_dir_only_path_ancestors_rev_2() {
    let path = Path::new(".");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next_back().unwrap().as_os_str(), OsStr::new(""));
    // next_back() should only see "." leftover
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("."));
    // We have consumed "" and ".", we should only observe
    // `None` being returned from either end
    assert_eq!(ancestors.next(), None);
    assert_eq!(ancestors.next_back(), None);

    // operates like next()
    let mut rev_ancestors = path.ancestors().rev();
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("."));

    // operates like next_back()
    let mut rev_ancestors = rev_ancestors.rev();
    assert_eq!(rev_ancestors.next().unwrap().as_os_str(), OsStr::new(""));

    // fully consumed, should return None
    let mut rev_ancestors = rev_ancestors.rev();
    assert_eq!(rev_ancestors.next_back(), None);
}

#[test]
fn single_letter_path_ancestors() {
    let path = Path::new("a");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("a"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(ancestors.next(), None);

    let mut rev_ancestors = path.ancestors();
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("a"));
    assert_eq!(rev_ancestors.next_back(), None);
}

#[test]
fn single_letter_trailing_path_ancestors() {
    let path = Path::new("a/");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("a/"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(ancestors.next(), None);

    let mut rev_ancestors = path.ancestors();
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("a/"));
    assert_eq!(rev_ancestors.next_back(), None);
}

#[test]
fn curr_dir_relative_path_ancestors() {
    let path = Path::new("./foo/bar");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("./foo/bar"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("./foo"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("."));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(ancestors.next(), None);

    let mut rev_ancestors = path.ancestors();
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("."));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("./foo"));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("./foo/bar"));
    assert_eq!(rev_ancestors.next_back(), None);
}

#[test]
fn parent_dir_only_path_ancestors() {
    let path = Path::new("..");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(".."));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(ancestors.next(), None);

    let mut rev_ancestors = path.ancestors();
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new(".."));
    assert_eq!(rev_ancestors.next_back(), None);
}

#[test]
fn parent_dir_relative_path_ancestors() {
    let path = Path::new("../foo/bar/");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("../foo/bar/"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("../foo"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(".."));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(ancestors.next(), None);

    let mut rev_ancestors = path.ancestors();
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new(".."));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("../foo"));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("../foo/bar/"));
    assert_eq!(rev_ancestors.next_back(), None);
}

#[test]
fn relative_path_ancestors() {
    let path = Path::new("foo/bar/baz/");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("foo/bar/baz/"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("foo/bar"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("foo"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(ancestors.next(), None);

    let mut rev_ancestors = path.ancestors();
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("foo"));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("foo/bar"));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("foo/bar/baz/"));
    assert_eq!(rev_ancestors.next_back(), None);
}

#[test]
fn one_letter_relative_path_ancestors() {
    let path = Path::new("a/");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("a/"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(ancestors.next(), None);

    let mut rev_ancestors = path.ancestors();
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new(""));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("a/"));
    assert_eq!(rev_ancestors.next_back(), None);
}

#[test]
fn root_dir_only_path_ancestors() {
    let path = Path::new("/");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("/"));
    assert_eq!(ancestors.next(), None);

    let mut rev_ancestors = path.ancestors();
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("/"));
    assert_eq!(rev_ancestors.next_back(), None);
}

#[test]
fn root_dir_trailing_path_ancestors() {
    let path = Path::new("////");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("////"));
    assert_eq!(ancestors.next(), None);

    let mut rev_ancestors = path.ancestors();
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("////"));
    assert_eq!(rev_ancestors.next_back(), None);
}

#[test]
fn absolute_path_ancestors() {
    let path = Path::new("/foo/bar/");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("/foo/bar/"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("/foo"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("/"));
    assert_eq!(ancestors.next(), None);

    let mut rev_ancestors = path.ancestors();
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("/"));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("/foo"));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("/foo/bar/"));
    assert_eq!(rev_ancestors.next_back(), None);
}

#[test]
fn absolute_with_in_between_trailing_seps_path_ancestors() {
    let path = Path::new("/foo/////bar/");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("/foo/////bar/"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("/foo"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("/"));
    assert_eq!(ancestors.next(), None);

    let mut rev_ancestors = path.ancestors();
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("/"));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("/foo"));
    assert_eq!(rev_ancestors.next_back().unwrap().as_os_str(), OsStr::new("/foo/////bar/"));
    assert_eq!(rev_ancestors.next_back(), None);
}

#[test]
fn absolute_rev_path_ancestors() {
    let path = Path::new("/foo/bar/baz/");
    let mut ancestors = path.ancestors();

    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("/foo/bar/baz/"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new("/foo/bar"));
    assert_eq!(ancestors.next_back().unwrap().as_os_str(), OsStr::new("/"));
    assert_eq!(ancestors.next_back().unwrap().as_os_str(), OsStr::new("/foo"));
    assert_eq!(ancestors.next(), None); // Fully consumed
    assert_eq!(ancestors.next_back(), None); // Fully consumed
}

#[cfg(windows)]
#[test]
fn verbatim_prefix_component_path_ancestors() {
    let path = Path::new(r"\\\\?\\UNC\\server\\share\\..");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(r"\\\\?\\UNC\\server\\share\\.."));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(r"\\\\?\\UNC\\server\\share\\"));
    assert_eq!(ancestors.next(), None);

    let mut ancestors = path.ancestors();
    assert_eq!(
        ancestors.next_back().unwrap().as_os_str(),
        OsStr::new(r"\\\\?\\UNC\\server\\share\\")
    );
    assert_eq!(
        ancestors.next_back().unwrap().as_os_str(),
        OsStr::new(r"\\\\?\\UNC\\server\\share\\..")
    );
    assert_eq!(ancestors.next_back(), None);
}

#[cfg(windows)]
#[test]
fn verbatim_unc_prefix_component_path_ancestors() {
    let path = Path::new(r"\\?\pictures\kittens");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(r"\\?\pictures\kittens"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(r"\\?\pictures"));
    assert_eq!(ancestors.next(), None);

    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next_back().unwrap().as_os_str(), OsStr::new(r"\\?\pictures\"));
    assert_eq!(ancestors.next_back().unwrap().as_os_str(), OsStr::new(r"\\?\pictures\kittens"));
    assert_eq!(ancestors.next_back(), None);
}

#[cfg(windows)]
#[test]
fn verbatim_disk_prefix_component_path_ancestors() {
    let path = Path::new(r"\\?\c:\Test");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(r"\\?\c:\Test"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(r"\\?\c:\"));
    assert_eq!(ancestors.next(), None);

    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next_back().unwrap().as_os_str(), OsStr::new(r"\\?\c:\"));
    assert_eq!(ancestors.next_back().unwrap().as_os_str(), OsStr::new(r"\\?\c:\Test"));
    assert_eq!(ancestors.next_back(), None);
}

#[cfg(windows)]
#[test]
fn device_ns_prefix_component_path_ancestors() {
    // No this will not execute notepad.exe
    let path = Path::new(r"\\.\c:\Windows\System32\notepad.exe");
    let mut ancestors = path.ancestors();
    assert_eq!(
        ancestors.next().unwrap().as_os_str(),
        OsStr::new(r"\\.\c:\Windows\System32\notepad.exe")
    );
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(r"\\.\c:\Windows\System32"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(r"\\.\c:\Windows"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(r"\\.\c:\"));
    assert_eq!(ancestors.next(), None);

    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next_back().unwrap().as_os_str(), OsStr::new(r"\\.\c:"));
    assert_eq!(ancestors.next_back().unwrap().as_os_str(), OsStr::new(r"\\.\c:\Windows"));
    assert_eq!(ancestors.next_back().unwrap().as_os_str(), OsStr::new(r"\\.\c:\Windows\System32"));
    assert_eq!(
        ancestors.next_back().unwrap().as_os_str(),
        OsStr::new(r"\\.\c:\Windows\System32\notepad.exe")
    );
    assert_eq!(ancestors.next_back(), None);
}

#[cfg(windows)]
#[test]
fn unc_prefix_component_path_ancestors() {
    let path = Path::new(r"\\server\share\test");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(r"\\server\share\test"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(r"\\server\share"));
    assert_eq!(ancestors.next(), None);

    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next_back().unwrap().as_os_str(), OsStr::new(r"\\server\share"));
    assert_eq!(ancestors.next_back().unwrap().as_os_str(), OsStr::new(r"\\server\share\test"));
    assert_eq!(ancestors.next_back(), None);
}

#[cfg(windows)]
#[test]
fn disk_prefix_component_path_ancestors() {
    let path = Path::new(r"C:a\..\..");
    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(r"C:a\..\.."));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(r"C:a\.."));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(r"C:a"));
    assert_eq!(ancestors.next().unwrap().as_os_str(), OsStr::new(r"C:"));
    assert_eq!(ancestors.next(), None);

    let mut ancestors = path.ancestors();
    assert_eq!(ancestors.next_back().unwrap().as_os_str(), OsStr::new(r"C:"));
    assert_eq!(ancestors.next_back().unwrap().as_os_str(), OsStr::new(r"C:a"));
    assert_eq!(ancestors.next_back().unwrap().as_os_str(), OsStr::new(r"C:a\.."));
    assert_eq!(ancestors.next_back().unwrap().as_os_str(), OsStr::new(r"C:a\..\.."));
    assert_eq!(ancestors.next_back(), None);
}
