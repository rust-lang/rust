//@ignore-target: windows # Supported only on unixes

fn test_sysconfbasic() {
    unsafe {
        let ncpus = libc::sysconf(libc::_SC_NPROCESSORS_CONF);
        assert!(ncpus >= 1);
        let psz = libc::sysconf(libc::_SC_PAGESIZE);
        assert!(psz % 4096 == 0);
        // note that in reality it can return -1 (no hard limit) on some platforms.
        let gwmax = libc::sysconf(libc::_SC_GETPW_R_SIZE_MAX);
        assert!(gwmax >= 512);
        let omax = libc::sysconf(libc::_SC_OPEN_MAX);
        assert_eq!(omax, 65536);
    }
}

fn main() {
    test_sysconfbasic();
}
