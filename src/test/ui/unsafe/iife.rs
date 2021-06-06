// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

unsafe fn u() {}

fn main() {
    u(); //~ ERROR call to unsafe function is unsafe

    (|| {
        u(); //~ ERROR call to unsafe function is unsafe
        unsafe {
            u();
        }
    })();

    unsafe {
        (|| {
            u();
        })();
    }

    {
        (|| {
            {
                u(); //~ ERROR call to unsafe function is unsafe
            }
        })();
    }

    unsafe {
        {
            (|| {
                {
                    u();
                }
            })();
        }
    }
}
