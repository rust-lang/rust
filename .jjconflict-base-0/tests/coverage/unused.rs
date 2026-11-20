#![allow(dead_code, unused_assignments, unused_must_use, unused_variables)]

fn foo<T>(x: T) {
    let mut i = 0;
    while i < 10 {
        i != 0 || i != 0;
        i += 1;
    }
}

fn unused_template_func<T>(x: T) {
    let mut i = 0;
    while i < 10 {
        i != 0 || i != 0;
        i += 1;
    }
}

fn unused_func(mut a: u32) {
    if a != 0 {
        a += 1;
    }
}

fn unused_func2(mut a: u32) {
    if a != 0 {
        a += 1;
    }
}

fn unused_func3(mut a: u32) {
    if a != 0 {
        a += 1;
    }
}

fn main() -> Result<(), u8> {
    foo::<u32>(0);
    foo::<f32>(0.0);
    Ok(())
}
