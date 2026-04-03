use syscalls::*;

#[test]
fn test_syscall() {
    let s = "Hello\0";
    assert_eq!(
        unsafe { syscall!(Sysno::write, 1, s.as_ptr() as *const _, 6) },
        Ok(6)
    );
}

#[test]
fn test_syscall_map() {
    // Make sure the macro exports are ok
    let mut map = SysnoMap::new();
    assert!(map.is_empty());
    assert_eq!(map.count(), 0);
    assert_eq!(map.get(Sysno::write), None);
    map.insert(Sysno::write, 42);
    assert_eq!(map.get(Sysno::write), Some(&42));
    assert_eq!(map.count(), 1);
    assert!(!map.is_empty());
    map.remove(Sysno::write);
    assert_eq!(map.count(), 0);
    assert!(map.is_empty());
}
