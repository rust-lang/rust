//! Check that if there are a lot of errors we truncate the list of errors appropriately

enum Question {
    Yes,
    No,
}

mod foo {
    const FOO: u32 = 0;
}

fn main() {
    let x = "a";
    x += 2;
    //~^ ERROR E0368
    y = 2;
    //~^ ERROR E0425
    x.z();
    //~^ ERROR E0599

    !Question::Yes;
    //~^ ERROR E0600

    foo::FOO;
    //~^ ERROR E0603

    0u32 as char;
    //~^ ERROR E0604

    let x = 0u8;
    x as Vec<u8>;
    //~^ ERROR E0605

    let x = 5;
    let x_is_nonzero = x as bool;
    //~^ ERROR E0054

    let x = &0u8;
    let y: u32 = x as u32;
    //~^ ERROR E0606

    let v = core::ptr::null::<u8>();
    v as *const [u8];
    //~^ ERROR E0607
}
