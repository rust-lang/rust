//@only-target: darwin

use std::thread;

extern "C" {
    fn _tlv_atexit(dtor: unsafe extern "C" fn(*mut u8), arg: *mut u8);
}

fn register<F>(f: F)
where
    F: FnOnce() + 'static,
{
    // This will receive the pointer passed into `_tlv_atexit`, which is the
    // original `f` but boxed up.
    unsafe extern "C" fn run<F>(ptr: *mut u8)
    where
        F: FnOnce() + 'static,
    {
        let f = unsafe { Box::from_raw(ptr as *mut F) };
        f()
    }

    unsafe {
        _tlv_atexit(run::<F>, Box::into_raw(Box::new(f)) as *mut u8);
    }
}

fn main() {
    thread::spawn(|| {
        register(|| println!("dtor 2"));
        register(|| println!("dtor 1"));
        println!("exiting thread");
    })
    .join()
    .unwrap();

    println!("exiting main");
    register(|| println!("dtor 5"));
    register(|| {
        println!("registering dtor in dtor 3");
        register(|| println!("dtor 4"));
    });
}
