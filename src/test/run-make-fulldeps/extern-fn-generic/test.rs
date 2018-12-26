extern crate testcrate;

extern "C" fn bar<T>(ts: testcrate::TestStruct<T>) -> T { ts.y }

#[link(name = "test", kind = "static")]
extern {
    fn call(c: extern "C" fn(testcrate::TestStruct<i32>) -> i32) -> i32;
}

fn main() {
    // Let's test calling it cross crate
    let back = unsafe {
        testcrate::call(testcrate::foo::<i32>)
    };
    assert_eq!(3, back);

    // And just within this crate
    let back = unsafe {
        call(bar::<i32>)
    };
    assert_eq!(3, back);
}
