//@ run-pass

static mut S: *const u8 = unsafe { &S as *const *const u8 as *const u8 };
//~^ WARN shared reference to mutable static [static_mut_refs]

struct StaticDoubleLinked {
    prev: &'static StaticDoubleLinked,
    next: &'static StaticDoubleLinked,
    data: i32,
    head: bool,
}

static L1: StaticDoubleLinked = StaticDoubleLinked { prev: &L3, next: &L2, data: 1, head: true };
static L2: StaticDoubleLinked = StaticDoubleLinked { prev: &L1, next: &L3, data: 2, head: false };
static L3: StaticDoubleLinked = StaticDoubleLinked { prev: &L2, next: &L1, data: 3, head: false };

pub fn main() {
    unsafe {
        assert_eq!(S, *(S as *const *const u8));
        //~^ WARN creating a shared reference to mutable static [static_mut_refs]
    }

    let mut test_vec = Vec::new();
    let mut cur = &L1;
    loop {
        test_vec.push(cur.data);
        cur = cur.next;
        if cur.head {
            break;
        }
    }
    assert_eq!(&test_vec, &[1, 2, 3]);

    let mut test_vec = Vec::new();
    let mut cur = &L1;
    loop {
        cur = cur.prev;
        test_vec.push(cur.data);
        if cur.head {
            break;
        }
    }
    assert_eq!(&test_vec, &[3, 2, 1]);
}
