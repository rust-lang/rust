#[repr(C)]
#[derive(Debug)]
struct PairFoo {
    fst: Foo,
    snd: Foo,
}

#[derive(Debug)]
struct Foo(u64);
fn reinterstruct(box_pair: Box<PairFoo>) -> Vec<Foo> {
    let ref_pair = Box::leak(box_pair) as *mut PairFoo;
    let ptr_foo = unsafe { &mut (*ref_pair).fst as *mut Foo };
    unsafe {
        Vec::from_raw_parts(ptr_foo, 2, 2)
    }
}

fn main() {
    let pair_foo = Box::new(PairFoo {
        fst: Foo(42),
        snd: Foo(1337),
    });
    println!("pair_foo = {:?}", pair_foo);
    for (n, foo) in reinterstruct(pair_foo).into_iter().enumerate() {
        println!("foo #{} = {:?}", n, foo);
    }
}

