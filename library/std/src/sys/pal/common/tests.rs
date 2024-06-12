use core::iter::repeat;

use crate::ffi::CString;
use crate::hint::black_box;
use crate::path::Path;
use crate::sys::common::small_c_string::run_path_with_cstr;

#[test]
fn stack_allocation_works() {
    let path = Path::new("abc");
    let result = run_path_with_cstr(path, &|p| {
        assert_eq!(p, &*CString::new(path.as_os_str().as_encoded_bytes()).unwrap());
        Ok(42)
    });
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn stack_allocation_fails() {
    let path = Path::new("ab\0");
    assert!(run_path_with_cstr::<()>(path, &|_| unreachable!()).is_err());
}

#[test]
fn heap_allocation_works() {
    let path = repeat("a").take(384).collect::<String>();
    let path = Path::new(&path);
    let result = run_path_with_cstr(path, &|p| {
        assert_eq!(p, &*CString::new(path.as_os_str().as_encoded_bytes()).unwrap());
        Ok(42)
    });
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn heap_allocation_fails() {
    let mut path = repeat("a").take(384).collect::<String>();
    path.push('\0');
    let path = Path::new(&path);
    assert!(run_path_with_cstr::<()>(path, &|_| unreachable!()).is_err());
}

#[bench]
fn bench_stack_path_alloc(b: &mut test::Bencher) {
    let path = repeat("a").take(383).collect::<String>();
    let p = Path::new(&path);
    b.iter(|| {
        run_path_with_cstr(p, &|cstr| {
            black_box(cstr);
            Ok(())
        })
        .unwrap();
    });
}

#[bench]
fn bench_heap_path_alloc(b: &mut test::Bencher) {
    let path = repeat("a").take(384).collect::<String>();
    let p = Path::new(&path);
    b.iter(|| {
        run_path_with_cstr(p, &|cstr| {
            black_box(cstr);
            Ok(())
        })
        .unwrap();
    });
}
