//@ run-pass
// check raw fat pointer ops

use std::mem;

fn assert_inorder<T: PartialEq + PartialOrd>(a: &[T]) {
    for i in 0..a.len() {
        for j in 0..a.len() {
            if i < j {
                assert!(a[i] < a[j]);
                assert!(a[i] <= a[j]);
                assert!(!(a[i] == a[j]));
                assert!(a[i] != a[j]);
                assert!(!(a[i] >= a[j]));
                assert!(!(a[i] > a[j]));
            } else if i == j {
                assert!(!(a[i] < a[j]));
                assert!(a[i] <= a[j]);
                assert!(a[i] == a[j]);
                assert!(!(a[i] != a[j]));
                assert!(a[i] >= a[j]);
                assert!(!(a[i] > a[j]));
            } else {
                assert!(!(a[i] < a[j]));
                assert!(!(a[i] <= a[j]));
                assert!(!(a[i] == a[j]));
                assert!(a[i] != a[j]);
                assert!(a[i] >= a[j]);
                assert!(a[i] > a[j]);
            }
        }
    }
}

trait Foo { fn foo(&self) -> usize; } //~ WARN method `foo` is never used
impl<T> Foo for T {
    fn foo(&self) -> usize {
        mem::size_of::<T>()
    }
}

#[allow(dead_code)]
struct S<T:?Sized>(u32, T);

fn main() {
    let mut array = [0,1,2,3,4];
    let mut array2 = [5,6,7,8,9];

    // fat ptr comparison: addr then extra

    // check ordering for arrays
    let mut ptrs: Vec<*const [u8]> = vec![
        &array[0..0], &array[0..1], &array, &array[1..]
    ];

    let array_addr = &array as *const [u8] as *const u8 as usize;
    let array2_addr = &array2 as *const [u8] as *const u8 as usize;
    if array2_addr < array_addr {
        ptrs.insert(0, &array2);
    } else {
        ptrs.push(&array2);
    }
    assert_inorder(&ptrs);

    // check ordering for mut arrays
    let mut ptrs: Vec<*mut [u8]> = vec![
        &mut array[0..0], &mut array[0..1], &mut array, &mut array[1..]
    ];

    let array_addr = &mut array as *mut [u8] as *mut u8 as usize;
    let array2_addr = &mut array2 as *mut [u8] as *mut u8 as usize;
    if array2_addr < array_addr {
        ptrs.insert(0, &mut array2);
    } else {
        ptrs.push(&mut array2);
    }
    assert_inorder(&ptrs);

    let mut u8_ = (0u8, 1u8);
    let mut u32_ = (4u32, 5u32);

    // check ordering for ptrs
    let buf: &mut [*const dyn Foo] = &mut [
        &u8_, &u8_.0,
        &u32_, &u32_.0,
    ];
    buf.sort_by(|u,v| {
        let u : [*const (); 2] = unsafe { mem::transmute(*u) };
        let v : [*const (); 2] = unsafe { mem::transmute(*v) };
        u.cmp(&v)
    });
    assert_inorder(buf);

    // check ordering for mut ptrs
    let buf: &mut [*mut dyn Foo] = &mut [
        &mut u8_, &mut u8_.0,
        &mut u32_, &mut u32_.0,
    ];
    buf.sort_by(|u,v| {
        let u : [*const (); 2] = unsafe { mem::transmute(*u) };
        let v : [*const (); 2] = unsafe { mem::transmute(*v) };
        u.cmp(&v)
    });
    assert_inorder(buf);

    // check ordering for structs containing arrays
    let ss: (S<[u8; 2]>,
             S<[u8; 3]>,
             S<[u8; 2]>) = (
        S(7, [8, 9]),
        S(10, [11, 12, 13]),
        S(4, [5, 6])
    );
    assert_inorder(&[
        &ss.0 as *const S<[u8]>,
        &ss.1 as *const S<[u8]>,
        &ss.2 as *const S<[u8]>
    ]);
}
