use crate::sys::pal::unix::fs::FilePermissions;

#[test]
fn test_debug_permissions() {
    for (expected, mode) in [
        // typical directory
        ("FilePermissions { mode: 0o040775 (drwxrwxr-x) }", 0o04_0775),
        // typical text file
        ("FilePermissions { mode: 0o100664 (-rw-rw-r--) }", 0o10_0664),
        // setuid executable (/usr/bin/doas)
        ("FilePermissions { mode: 0o104755 (-rwsr-xr-x) }", 0o10_4755),
        // char device (/dev/zero)
        ("FilePermissions { mode: 0o020666 (crw-rw-rw-) }", 0o02_0666),
        // block device (/dev/vda)
        ("FilePermissions { mode: 0o060660 (brw-rw----) }", 0o06_0660),
        // symbolic link
        ("FilePermissions { mode: 0o120777 (lrwxrwxrwx) }", 0o12_0777),
        // fifo
        ("FilePermissions { mode: 0o010664 (prw-rw-r--) }", 0o01_0664),
        // none
        ("FilePermissions { mode: 0o100000 (----------) }", 0o10_0000),
        // unrecognized
        ("FilePermissions { mode: 0o000001 }", 1),
    ] {
        assert_eq!(format!("{:?}", FilePermissions { mode }), expected);
    }

    for (expected, mode) in [
        // owner readable
        ("FilePermissions { mode: 0o100400 (-r--------) }", libc::S_IRUSR),
        // owner writable
        ("FilePermissions { mode: 0o100200 (--w-------) }", libc::S_IWUSR),
        // owner executable
        ("FilePermissions { mode: 0o100100 (---x------) }", libc::S_IXUSR),
        // setuid
        ("FilePermissions { mode: 0o104000 (---S------) }", libc::S_ISUID),
        // owner executable and setuid
        ("FilePermissions { mode: 0o104100 (---s------) }", libc::S_IXUSR | libc::S_ISUID),
        // group readable
        ("FilePermissions { mode: 0o100040 (----r-----) }", libc::S_IRGRP),
        // group writable
        ("FilePermissions { mode: 0o100020 (-----w----) }", libc::S_IWGRP),
        // group executable
        ("FilePermissions { mode: 0o100010 (------x---) }", libc::S_IXGRP),
        // setgid
        ("FilePermissions { mode: 0o102000 (------S---) }", libc::S_ISGID),
        // group executable and setgid
        ("FilePermissions { mode: 0o102010 (------s---) }", libc::S_IXGRP | libc::S_ISGID),
        // other readable
        ("FilePermissions { mode: 0o100004 (-------r--) }", libc::S_IROTH),
        // other writeable
        ("FilePermissions { mode: 0o100002 (--------w-) }", libc::S_IWOTH),
        // other executable
        ("FilePermissions { mode: 0o100001 (---------x) }", libc::S_IXOTH),
        // sticky
        ("FilePermissions { mode: 0o101000 (----------) }", libc::S_ISVTX),
        // other executable and sticky
        ("FilePermissions { mode: 0o101001 (---------x) }", libc::S_IXOTH | libc::S_ISVTX),
    ] {
        assert_eq!(format!("{:?}", FilePermissions { mode: libc::S_IFREG | mode }), expected);
    }

    for (expected, mode) in [
        // restricted deletion ("sticky") flag is set, and search permission is not granted to others
        ("FilePermissions { mode: 0o041000 (d--------T) }", libc::S_ISVTX),
        // sticky and searchable
        ("FilePermissions { mode: 0o041001 (d--------t) }", libc::S_ISVTX | libc::S_IXOTH),
    ] {
        assert_eq!(format!("{:?}", FilePermissions { mode: libc::S_IFDIR | mode }), expected);
    }
}
